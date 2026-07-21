import os
import sys
import glob
import logging
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
from itertools import chain
from abc import ABC, abstractmethod

# Force stdout/stderr to use UTF-8 encoding to prevent UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Configure Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from config import SplitterConfig


class BaseDataSplitter(ABC):
    """
    Abstract base class defining the dataset splitting interface and shared utilities.
    Handles common split export logic (CSV, JSONL, summary JSON) for all splitter implementations.
    """
    def __init__(self, config: SplitterConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset_path
        self.seed = config.seed
        self.test_sample_per_class = config.test_sample_per_class
        self.save_results = config.save_results
        self.include_video = config.include_video

        # Automatically resolve the audio directory path (used as the root for sample splitting)
        audio_dir = os.path.join(self.dataset_path, 'audio')
        if os.path.isdir(audio_dir):
            self.audio_path = audio_dir
        else:
            self.audio_path = self.dataset_path

        # Automatically resolve the video directory path
        video_dir = os.path.join(self.dataset_path, 'video')
        if os.path.isdir(video_dir):
            self.video_path = video_dir
            self.video_exists = True
        else:
            self.video_path = None
            self.video_exists = False
            # If video was requested but no video directory exists, log a warning and auto-fallback
            if self.include_video:
                logger.warning("==================================================")
                logger.warning("Cảnh báo: Không tìm thấy thư mục 'video' trong dataset_path.")
                logger.warning("Tự động chuyển chế độ RAM output về chỉ bao gồm đường dẫn 'audio'.")
                logger.warning("==================================================")
                self.include_video = False

    @abstractmethod
    def get_file_list(self, split_name: str) -> List[str]:
        """
        Scan the directory and return a list of file paths.
        Must be implemented by subclasses.

        Args:
            split_name (str): The feeding intensity class name to scan.
                Example: 'strong', 'medium', 'weak', 'none'

        Returns:
            List[str]: A list of absolute paths of all scanned files.
        """
        pass

    @abstractmethod
    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        """
        Split the file lists into Train, Test, and Validation sets.
        Must be implemented by subclasses.

        Args:
            None (parameters are derived from class attributes).

        Returns:
            Tuple[List[List], List[List], List[List]]: A tuple containing three lists (Train, Test, Val).
        """
        pass

    @abstractmethod
    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        """
        Format raw data lists into standardized dictionaries for file export.
        Must be implemented by subclasses.

        Args:
            split_name (str): The dataset split name ('train', 'test', or 'val').
            data_list (List[List]): A list of label-paired data entries in [path, numeric_label] format.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing detailed metadata for each sample.
        """
        pass

    @staticmethod
    def _write_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
        """Write a list of data samples to a JSONL-formatted file."""
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_csv(samples: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
        """Write a list of data samples to a CSV-formatted file."""
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

    def _save_splits(self, train_dict: List[List], test_dict: List[List], val_dict: List[List], output_dir: Path) -> None:
        """
        Save dataset splits to CSV, JSONL, and a summary JSON file (shared implementation).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": train_dict,
            "test": test_dict,
            "val": val_dict
        }

        formatted_splits: Dict[str, List[Dict[str, Any]]] = {}

        # 1. Format data and save CSV/JSONL for each split using the subclass implementation
        for split_name, data_list in splits.items():
            samples = self._format_samples(split_name, data_list)
            formatted_splits[split_name] = samples

            # Proceed to write output files
            fieldnames = ["video_path", "audio_path", "label", "class_name", "date", "session", "sample_id"]
            self._write_jsonl(samples, output_path / f"{split_name}.jsonl")
            self._write_csv(samples, output_path / f"{split_name}.csv", fieldnames=fieldnames)
            logger.info(f"Đã lưu thành công các file phân chia vào {output_path}")

        # 2. Generate and save the summary.json file
        summary: Dict[str, Any] = {
            "label_map": {
                "none": 0,
                "strong": 1,
                "medium": 2,
                "weak": 3
              },
            "splits": {}
        }
        
        for split_name, samples in formatted_splits.items():
            counts = {"none": 0, "strong": 0, "medium": 0, "weak": 0}
            for s in samples:
                counts[str(s["class_name"])] += 1
            summary["splits"][split_name] = {
                "total": len(samples),
                "by_class": counts
            }

        # Append configuration metadata
        summary.update({
            "dataset_root": self.dataset_path,
            "output_dir": str(output_path),
            "seed": self.seed,
            "test_sample_per_class": self.test_sample_per_class
        })

        with (output_path / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu thành công file tóm tắt phân chia vào {output_path}")


class FishDataSplitter(BaseDataSplitter):
    """
    Concrete implementation for multimodal fish data splitting (Audio + Video).
    Preserves the original file scanning and splitting logic from U-FFIA's fish_audio_dataset.py.
    Automatically links video to audio and applies path-fixing fallback for cross-machine transfers.
    """
    # noinspection DuplicatedCode
    def get_file_list(self, split_name: str) -> List[str]:
        # Preserve the original get_wav_name logic from U-FFIA source code for raw data scanning
        path = self.audio_path
        audio = []
        l1 = os.listdir(path)
        for folder_name in l1:
            l2 = os.listdir(os.path.join(path, folder_name))
            for session_folder in l2:
                wav_dir = os.path.join(path, folder_name, session_folder, split_name, '*.wav')
                audio.append(glob.glob(wav_dir))
        return list(chain.from_iterable(audio))

    def _resolve_video_path(self, audio_path: str) -> str:
        """Automatically resolve the corresponding video path from an audio path."""
        # Replace backslashes with forward slashes to ensure cross-platform compatibility
        normalized_audio = str(audio_path).replace('\\', '/')
        parts = Path(normalized_audio).parts
        parts_list = list(parts)
        if "audio" in parts_list:
            idx = parts_list.index("audio")
            parts_list[idx] = "video"
        
        filename = parts_list[-1]
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".mp4"
        if "_audio_" in filename:
            filename = filename.replace("_audio_", "_video_")
        parts_list[-1] = filename
        
        if parts_list[0].endswith('\\') or parts_list[0].endswith('/'):
            reconstructed_video = parts_list[0] + os.path.join(*parts_list[1:])
        else:
            reconstructed_video = os.path.join(*parts_list)

        if os.path.exists(reconstructed_video):
            return reconstructed_video
        return ""

    def split_data(self) -> Tuple[List[List], List[List], List[List]]:
        # Automatically determine the splits directory
        if Path(self.dataset_path).name in ['audio', 'video']:
            splits_dir = Path(self.dataset_path).parent / 'splits'
        else:
            splits_dir = Path(self.dataset_path) / 'splits'

        train_csv = splits_dir / 'train.csv'
        test_csv = splits_dir / 'test.csv'
        val_csv = splits_dir / 'val.csv'

        if splits_dir.exists() and train_csv.exists() and test_csv.exists() and val_csv.exists():
            logger.info("==================================================")
            logger.info(f"Kích hoạt chế độ Fallback: Phát hiện phân chia có sẵn tại '{splits_dir}'.")
            logger.info("Đang nạp tập dữ liệu đã chia thay vì tính toán lại...")
            logger.info(f"Đường dẫn tìm kiếm âm thanh cơ sở: '{self.audio_path}'")
            logger.info("==================================================")

            train_dict = []
            test_dict = []
            val_dict = []
            need_rewrite_files = False

            try:
                # Helper function to load data from CSV and auto-fix file paths with batched logging
                def load_and_fix_csv(csv_path: Path, split_name: str) -> List[List]:
                    nonlocal need_rewrite_files
                    loaded_data = []
                    fixed_count = 0
                    missing_count = 0
                    video_updated_count = 0
                    
                    # Variables for logging a few example path corrections
                    audio_fix_example = None
                    video_fix_example = None
                    
                    with csv_path.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            raw_audio_path = row["audio_path"]
                            label = int(row["label"])
                            
                            # If the audio path does not exist, attempt automatic path correction
                            if not os.path.exists(raw_audio_path):
                                # Replace backslashes with forward slashes to ensure cross-platform compatibility
                                normalized_audio = raw_audio_path.replace('\\', '/')
                                parts = Path(normalized_audio).parts
                                if len(parts) >= 4:
                                    # Generate multiple possible paths due to different server layouts
                                    candidates = [
                                        os.path.join(self.audio_path, parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(self.dataset_path, 'audio', parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(self.dataset_path, parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(self.dataset_path, 'U_FFIA', 'audio', parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(str(Path(self.dataset_path).parent), 'audio', parts[-4], parts[-3], parts[-2], parts[-1]),
                                        os.path.join(str(Path(self.dataset_path).parent), 'U_FFIA', 'audio', parts[-4], parts[-3], parts[-2], parts[-1]),
                                    ]
                                    
                                    fixed_path = None
                                    for cand in candidates:
                                        if os.path.exists(cand):
                                            fixed_path = cand
                                            break
                                            
                                    if fixed_path is not None:
                                        if audio_fix_example is None:
                                            audio_fix_example = (raw_audio_path, fixed_path)
                                        raw_audio_path = fixed_path
                                        fixed_count += 1
                                        need_rewrite_files = True
                                    else:
                                        if missing_count < 3:
                                            # Display the primary checked path in warning for reference
                                            primary_checked = candidates[0]
                                            logger.warning(f"Tập {split_name}: Thử các đường dẫn sửa đổi (ví dụ: '{primary_checked}') nhưng file vẫn không tồn tại trên server!")
                                        missing_count += 1
                            
                            # Reconstruct and verify the existence of video_path
                            video_path_str = row.get("video_path", "")
                            if video_path_str and not os.path.exists(video_path_str):
                                # Replace backslashes with forward slashes to ensure cross-platform compatibility
                                normalized_video = video_path_str.replace('\\', '/')
                                parts_v = Path(normalized_video).parts
                                if len(parts_v) >= 4:
                                    candidates_v = [
                                        os.path.join(self.dataset_path, 'video', parts_v[-4], parts_v[-3], parts_v[-2], parts_v[-1]),
                                        os.path.join(self.dataset_path, parts_v[-4], parts_v[-3], parts_v[-2], parts_v[-1]),
                                        os.path.join(self.dataset_path, 'U_FFIA', 'video', parts_v[-4], parts_v[-3], parts_v[-2], parts_v[-1]),
                                        os.path.join(str(Path(self.dataset_path).parent), 'video', parts_v[-4], parts_v[-3], parts_v[-2], parts_v[-1]),
                                        os.path.join(str(Path(self.dataset_path).parent), 'U_FFIA', 'video', parts_v[-4], parts_v[-3], parts_v[-2], parts_v[-1]),
                                    ]
                                    fixed_video = None
                                    for cand in candidates_v:
                                        if os.path.exists(cand):
                                            fixed_video = cand
                                            break
                                            
                                    if fixed_video is not None:
                                        if video_fix_example is None:
                                            video_fix_example = (video_path_str, fixed_video)
                                        video_path_str = fixed_video
                                        need_rewrite_files = True
                                    else:
                                        # Attempt to re-resolve the video path from the audio file
                                        video_path_str = self._resolve_video_path(raw_audio_path)
                            elif not video_path_str:
                                # Case where legacy split files have no video_path (empty "") but video folder now exists
                                resolved_v = self._resolve_video_path(raw_audio_path)
                                if resolved_v:
                                    if video_fix_example is None:
                                        video_fix_example = ("", resolved_v)
                                    video_path_str = resolved_v
                                    video_updated_count += 1
                                    need_rewrite_files = True

                            # Return format depends on the include_video configuration
                            if self.include_video:
                                loaded_data.append([raw_audio_path, video_path_str, label])
                            else:
                                loaded_data.append([raw_audio_path, label])
                    
                    # Print detailed logs
                    if fixed_count > 0:
                        logger.info(f"Tập {split_name}: Tự động phát hiện và sửa đổi thành công {fixed_count} đường dẫn audio sai lệch.")
                        if audio_fix_example:
                            logger.info(f"   * Ví dụ sửa đổi đường dẫn audio:\n     - Cũ: {audio_fix_example[0]}\n     - Mới: {audio_fix_example[1]}")
                    if video_updated_count > 0:
                        logger.info(f"Tập {split_name}: Tự động bổ sung thành công {video_updated_count} đường dẫn video mới phát hiện trên máy.")
                        if video_fix_example:
                            logger.info(f"   * Ví dụ bổ sung video path: {video_fix_example[1]}")
                    if missing_count > 0:
                        logger.warning(f"Tập {split_name}: Có {missing_count} tệp tin không tồn tại trên máy hiện tại (vui lòng kiểm tra lại bộ dữ liệu).")
                        
                    return loaded_data

                train_dict = load_and_fix_csv(train_csv, "Train")
                test_dict = load_and_fix_csv(test_csv, "Test")
                val_dict = load_and_fix_csv(val_csv, "Validation")

                logger.info(f"Đã nạp thành công các phân chia từ file:")
                logger.info(f"- Tổng số mẫu tập Train: {len(train_dict)}")
                logger.info(f"- Tổng số mẫu tập Test:  {len(test_dict)}")
                logger.info(f"- Tổng số mẫu tập Val:   {len(val_dict)}")
                
                # Print first samples for user verification
                if len(train_dict) > 0:
                    logger.info(f"  * Mẫu Train đầu tiên được nạp: {train_dict[0]}")
                if len(test_dict) > 0:
                    logger.info(f"  * Mẫu Test đầu tiên được nạp:  {test_dict[0]}")
                if len(val_dict) > 0:
                    logger.info(f"  * Mẫu Val đầu tiên được nạp:   {val_dict[0]}")
                
                # If any paths were corrected or new video paths added, overwrite split files on disk to synchronize
                if need_rewrite_files:
                    logger.info("Đang tự động cập nhật và ghi đè lại các tệp tin phân chia trên đĩa để đồng bộ các thay đổi...")
                    self._save_splits(train_dict, test_dict, val_dict, splits_dir)
                    logger.info("Đồng bộ và sửa đổi tệp tin phân chia trên đĩa hoàn tất!")
                
                logger.info("==================================================")
                return train_dict, test_dict, val_dict
            except Exception as e:
                logger.warning(f"Lỗi nạp hoặc tự động sửa phân chia: {e}. Chuyển sang quy trình phân chia tiêu chuẩn.")

        logger.info("==================================================")
        logger.info("Bắt đầu quá trình phân chia dữ liệu âm thanh và video...")
        logger.info(f"Thư mục gốc bộ dữ liệu: '{self.dataset_path}'")
        logger.info(f"Seed ngẫu nhiên: {self.seed}")
        logger.info(f"Số lượng mẫu Test/Val mỗi class: {self.test_sample_per_class}")
        logger.info(f"Tự động lưu kết quả: {self.save_results}")
        logger.info(f"RAM bao gồm video path: {self.include_video}")
        logger.info("==================================================")

        # Scan files and log the counts
        logger.info("Đang quét danh sách file âm thanh cho từng class...")
        strong_list = self.get_file_list(split_name='strong')
        logger.info(f"Class 'strong': Tìm thấy {len(strong_list)} files.")
        
        medium_list = self.get_file_list(split_name='medium')
        logger.info(f"Class 'medium': Found {len(medium_list)} files.")
        
        weak_list = self.get_file_list(split_name='weak')
        logger.info(f"Class 'weak': Tìm thấy {len(weak_list)} files.")
        
        none_list = self.get_file_list(split_name='none')
        logger.info(f"Class 'none': Tìm thấy {len(none_list)} files.")

        # Shuffle each class list independently
        logger.info(f"Đang tiến hành xáo trộn (shuffle) danh sách độc lập với seed={self.seed}...")
        random_state = np.random.RandomState(self.seed)
        random_state.shuffle(strong_list)
        random_state.shuffle(medium_list)
        random_state.shuffle(weak_list)
        random_state.shuffle(none_list)

        # Perform dataset splitting
        logger.info("Đang tiến hành phân chia (slicing) các tập dữ liệu...")
        strong_test = strong_list[:self.test_sample_per_class]
        medium_test = medium_list[:self.test_sample_per_class]
        weak_test = weak_list[:self.test_sample_per_class]
        none_test = none_list[:self.test_sample_per_class]

        strong_val = strong_list[self.test_sample_per_class:2*self.test_sample_per_class]
        medium_val = medium_list[self.test_sample_per_class:2*self.test_sample_per_class]
        weak_val = weak_list[self.test_sample_per_class:2*self.test_sample_per_class]
        none_val = none_list[self.test_sample_per_class:2*self.test_sample_per_class]

        strong_train = strong_list[2*self.test_sample_per_class:]
        medium_train = medium_list[2*self.test_sample_per_class:]
        weak_train = weak_list[2*self.test_sample_per_class:]
        none_train = none_list[2*self.test_sample_per_class:]

        # Warn if there are insufficient samples for splitting
        for class_name, size in [('strong', len(strong_list)), ('medium', len(medium_list)), ('weak', len(weak_list)), ('none', len(none_list))]:
            req_samples = 2 * self.test_sample_per_class
            if size < req_samples:
                logger.warning(f"Class '{class_name}' chỉ có {size} files, trong khi cần tối thiểu {req_samples} samples để chia Test và Val.")

        # Log detailed split sizes per class
        logger.info(f"Chi tiết phân chia mỗi class:")
        logger.info(f"       - class 'strong': Train={len(strong_train)}, Test={len(strong_test)}, Val={len(strong_val)}")
        logger.info(f"       - class 'medium': Train={len(medium_train)}, Test={len(medium_test)}, Val={len(medium_val)}")
        logger.info(f"       - class 'weak':   Train={len(weak_train)}, Test={len(weak_test)}, Val={len(weak_val)}")
        logger.info(f"       - class 'none':   Train={len(none_train)}, Test={len(none_test)}, Val={len(none_val)}")

        # Map integer labels and merge lists using list comprehension
        logger.info("Đang ánh xạ nhãn số nguyên và tạo danh sách tập dữ liệu...")
        if self.include_video:
            train_dict = (
                [[wav, self._resolve_video_path(wav), 1] for wav in strong_train] +
                [[wav, self._resolve_video_path(wav), 2] for wav in medium_train] +
                [[wav, self._resolve_video_path(wav), 3] for wav in weak_train] +
                [[wav, self._resolve_video_path(wav), 0] for wav in none_train]
            )
            test_dict = (
                [[wav, self._resolve_video_path(wav), 1] for wav in strong_test] +
                [[wav, self._resolve_video_path(wav), 2] for wav in medium_test] +
                [[wav, self._resolve_video_path(wav), 3] for wav in weak_test] +
                [[wav, self._resolve_video_path(wav), 0] for wav in none_test]
            )
            val_dict = (
                [[wav, self._resolve_video_path(wav), 1] for wav in strong_val] +
                [[wav, self._resolve_video_path(wav), 2] for wav in medium_val] +
                [[wav, self._resolve_video_path(wav), 3] for wav in weak_val] +
                [[wav, self._resolve_video_path(wav), 0] for wav in none_val]
            )
        else:
            train_dict = (
                [[wav, 1] for wav in strong_train] +
                [[wav, 2] for wav in medium_train] +
                [[wav, 3] for wav in weak_train] +
                [[wav, 0] for wav in none_train]
            )
            test_dict = (
                [[wav, 1] for wav in strong_test] +
                [[wav, 2] for wav in medium_test] +
                [[wav, 3] for wav in weak_test] +
                [[wav, 0] for wav in none_test]
            )
            val_dict = (
                [[wav, 1] for wav in strong_val] +
                [[wav, 2] for wav in medium_val] +
                [[wav, 3] for wav in weak_val] +
                [[wav, 0] for wav in none_val]
            )

        # Final shuffle of the Train set
        logger.info("Xáo trộn (shuffle) lần cuối cho tập dữ liệu Train...")
        random_state.shuffle(train_dict)

        logger.info("==================================================")
        logger.info("Phân chia hoàn tất thành công!")
        logger.info(f"- Tổng số mẫu tập Train: {len(train_dict)}")
        logger.info(f"- Tổng số mẫu tập Test:  {len(test_dict)}")
        logger.info(f"- Tổng số mẫu tập Val:   {len(val_dict)}")
        
        # Print first samples for user verification
        if len(train_dict) > 0:
            logger.info(f"  * Mẫu Train đầu tiên được sinh: {train_dict[0]}")
        if len(test_dict) > 0:
            logger.info(f"  * Mẫu Test đầu tiên được sinh:  {test_dict[0]}")
        if len(val_dict) > 0:
            logger.info(f"  * Mẫu Val đầu tiên được sinh:   {val_dict[0]}")
        logger.info("==================================================")

        # Automatically save results to the splits directory alongside dataset_path if save_results is True
        if self.save_results:
            self._save_splits(train_dict, test_dict, val_dict, splits_dir)

        return train_dict, test_dict, val_dict

    def _format_samples(self, split_name: str, data_list: List[List]) -> List[Dict[str, Any]]:
        """Parse sample information into standardized dictionaries for file export, with automatic video linking."""
        label_to_class = {0: "none", 1: "strong", 2: "medium", 3: "weak"}
        samples = []
        for item in data_list:
            # Support both formats: 2-element array [audio_path, label] or 3-element array [audio_path, video_path, label]
            if len(item) == 3:
                wav_path = item[0]
                video_path_str = item[1]
                label = item[2]
            else:
                wav_path = item[0]
                label = item[1]
                video_path_str = self._resolve_video_path(wav_path)

            # Extract date, session, and sample_id from the absolute path
            parts = Path(wav_path).parts
            date_part = parts[-4] if len(parts) >= 4 else ""
            session_part = parts[-3] if len(parts) >= 3 else ""
            stem = Path(wav_path).stem
            sample_id = stem.split("_audio_")[-1] if "_audio_" in stem else stem

            samples.append({
                "video_path": video_path_str,
                "audio_path": str(wav_path),
                "label": int(label),
                "class_name": label_to_class.get(label, ""),
                "date": date_part,
                "session": session_part,
                "sample_id": sample_id
            })
        return samples

# Main execution block for standalone usage reference
if __name__ == '__main__':
    # Usage example for FishDataSplitter in Python:
    # 
    # from config import TrainConfig, SplitterConfig
    # from dataset import FishDataSplitter
    # 
    # 1. Load splitter config from unified train_config.json:
    # config = TrainConfig.from_json('config/train_config.json')
    # splitter_cfg = config.dataset_splitter
    # 
    # 2. Initialize the data splitter:
    # splitter = FishDataSplitter(config=splitter_cfg)
    # 
    # 3. Split data and receive dataset lists in RAM:
    # train_data, test_data, val_data = splitter.split_data()
    pass
