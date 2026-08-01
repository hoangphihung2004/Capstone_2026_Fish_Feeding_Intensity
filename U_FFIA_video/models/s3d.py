import torch.nn as nn

try:
    from torchvision.models.video import S3D_Weights, s3d
except ImportError as exc:
    S3D_Weights = None
    s3d = None
    _S3D_IMPORT_ERROR = exc
else:
    _S3D_IMPORT_ERROR = None


class S3D(nn.Module):
    """Torchvision S3D video classifier with optional Kinetics-400 weights."""

    input_type = "clip"
    minimum_frames = 14

    def __init__(self, classes_num: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        if s3d is None:
            raise ImportError(
                "S3D requires a torchvision version that provides "
                "torchvision.models.video.s3d."
            ) from _S3D_IMPORT_ERROR

        weights = S3D_Weights.DEFAULT if pretrained else None
        self.model = s3d(weights=weights)

        classifier = self.model.classifier[1]
        self.model.classifier[1] = nn.Conv3d(
            in_channels=classifier.in_channels,
            out_channels=classes_num,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.model_name = "s3d"

    def get_name(self) -> str:
        return self.model_name

    def forward(self, x):
        return self.model(x)
