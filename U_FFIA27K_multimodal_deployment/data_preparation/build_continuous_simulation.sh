#!/usr/bin/env bash
# Rebuild the continuous stream used by the dashboard's real-time simulation:
# all 2-second clips of one session concatenated in the order
# none -> strong -> medium -> weak (2022_6_19 / AM_100: 29+90+75+105 = 299 clips, 598 s).
#
# Usage: data_preparation/build_continuous_simulation.sh <dataset_dir> <samples_dir> [date] [session]
set -euo pipefail

DATASET="${1:?dataset dir}"
SAMPLES="${2:?samples dir}"
DATE="${3:-2022_6_19}"
SESSION="${4:-AM_100}"
NAME="U_FFIA_${DATE}_${SESSION}_none_strong_medium_weak"
OUT="${SAMPLES}/continuous_simulation"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${OUT}"

for cls in none strong medium weak; do
  ls "${DATASET}/video/${DATE}/${SESSION}/${cls}"/*.mp4 | sort -V | sed "s/^/file '/; s/$/'/" >> "${TMP}/video.txt"
  ls "${DATASET}/audio/${DATE}/${SESSION}/${cls}"/*.wav | sort -V >> "${TMP}/audio_src.txt"
done
echo "clips: $(wc -l < "${TMP}/video.txt") video, $(wc -l < "${TMP}/audio_src.txt") audio"

# Source clips are MPEG-4 Part 2, which browsers cannot play in <video>.
# Re-encode to H.264 (yuv420p, native resolution) so the dashboard can show it.
# Orin Nano has no hardware encoder, so this runs on the CPU (~15-25 min).
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "${TMP}/video.txt" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart "${OUT}/${NAME}.mp4"

# Source clips mix 256 kHz and 64 kHz; normalise each clip to 64 kHz mono (the
# model's input rate) before concatenating, otherwise the stream drifts.
mkdir -p "${TMP}/a"
i=0
while read -r f; do
  i=$((i + 1))
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$f" -ac 1 -ar 64000 -c:a pcm_s16le "${TMP}/a/$(printf '%04d' "$i").wav"
  echo "file '${TMP}/a/$(printf '%04d' "$i").wav'" >> "${TMP}/audio.txt"
done < "${TMP}/audio_src.txt"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "${TMP}/audio.txt" -c copy "${OUT}/${NAME}.wav"

for f in "${OUT}/${NAME}.mp4" "${OUT}/${NAME}.wav"; do
  echo "$(basename "$f"): $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f") s"
done
