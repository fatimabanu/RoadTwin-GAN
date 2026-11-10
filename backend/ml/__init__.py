"""ML modules for RoadTwin-GAN preprocessing, GAN synthesis, detection, and evaluation."""

from ml.gan import RoadConditionalDiscriminator, RoadConditionalGenerator, generate_synthetic_images
from ml.detector import RoadDamageCNN, analyze_road_image
from ml.preprocessing import (
    decode_base64_image,
    decode_image_bytes,
    encode_image_base64,
    extract_road_features,
    preprocess_for_model,
)

__all__ = [
    "RoadConditionalDiscriminator",
    "RoadConditionalGenerator",
    "RoadDamageCNN",
    "analyze_road_image",
    "decode_base64_image",
    "decode_image_bytes",
    "encode_image_base64",
    "extract_road_features",
    "generate_synthetic_images",
    "preprocess_for_model",
]
