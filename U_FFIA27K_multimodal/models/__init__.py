from .multimodal_model import MultimodalModel
from .student_model import AudioStudentModel, StftMlpStudentModel
from .surgery import adapt_first_conv_to_multi_channels

__all__ = [
    "AudioStudentModel",
    "MultimodalModel",
    "StftMlpStudentModel",
    "adapt_first_conv_to_multi_channels",
]
