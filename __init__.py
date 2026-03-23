"""Пакет распознавания автомобильных номеров.

Пример использования:
    from plate_recognition import LicensePlateRecognition, is_valid_plate

    lpr = LicensePlateRecognition()
    result = await lpr.get_plates("image.jpg")
"""

from .detection import Detection
from .pipeline import LicensePlateRecognition, is_valid_plate
from .recognition import Recognition

__all__ = [
    "Detection",
    "Recognition",
    "LicensePlateRecognition",
    "is_valid_plate",
]
