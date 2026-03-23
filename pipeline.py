"""Основной модуль системы распознавания автомобильных номеров."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from .detection import Detection
from .recognition import Recognition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PLATE_PATTERN = re.compile(
    r"^([АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3})"
    r"|([АВЕКМНОРСТУХ]\d{4}[АВЕКМНОРСТУХ]{2}\d{2,3})$"
)


def is_valid_plate(text: str) -> bool:
    """
    Проверяет валидность распознанного номера по паттерну.

    Args:
        text: Распознанный текст (может содержать несколько номеров).

    Returns:
        True, если все номера валидны, иначе False.
    """
    list_of_plates = text.split()
    for plate in list_of_plates:
        if not bool(PLATE_PATTERN.match(plate)):
            return False
    return True


class LicensePlateRecognition:
    """Основной класс системы распознавания автомобильных номеров.

    Предоставляет асинхронный API для детекции и распознавания номеров.
    """

    def __init__(
        self,
        path_to_det: str = r"models\detection\detection-v2.pt",
        path_to_rec: str = r"models\recognition",
    ) -> None:
        """
        Инициализирует систему распознавания номеров.

        Args:
            path_to_det: Путь к модели детекции.
            path_to_rec: Путь к модели распознавания.
        """
        self.det: Optional[Detection] = None
        self.rec: Optional[Recognition] = None

        try:
            self.det = Detection(path_to_det)
            logger.info("Модель детекции загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели детекции: {e}")

        try:
            self.rec = Recognition(model_dir=path_to_rec)
            logger.info("Модель распознавания загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели распознавания: {e}")

    async def get_plates(self, path_to_image: str | Path) -> str:
        """
        Распознает автомобильные номера на изображении.

        Args:
            path_to_image: Путь к изображению.

        Returns:
            Распознанный текст номеров (через пробел, если несколько).
        """
        if self.det is None:
            logger.error("Модель детекции не инициализирована")
            return ""

        try:
            img_rois = await self.det.get_rois(path_to_image)
            if not img_rois:
                logger.warning("Номера не найдены")
                return ""
            logger.info(f"Найдено номеров: {len(img_rois)}")
        except Exception as e:
            logger.error(f"Ошибка на этапе детекции: {e}")
            return ""

        text = ""
        if self.rec is not None:
            try:
                text = await self.rec.get_plates(img_rois)
                logger.info(f"Распознанный текст: {text}")
            except Exception as e:
                logger.error(f"Ошибка распознавания: {e}")
        else:
            logger.error("Модель распознавания не инициализирована")

        return text

    async def get_plates_with_validation(
        self, path_to_image: str | Path
    ) -> dict:
        """
        Распознает номера и возвращает результат с валидацией.

        Args:
            path_to_image: Путь к изображению.

        Returns:
            Словарь с результатами:
                - plates: распознанный текст
                - valid: True если номера валидны
                - count: количество найденных номеров
        """
        plates = await self.get_plates(path_to_image)
        count = len(plates.split()) if plates else 0

        return {
            "plates": plates,
            "valid": is_valid_plate(plates) if plates else False,
            "count": count,
        }


async def main() -> None:
    """Пример использования системы распознавания."""
    path_to_image = r""  # путь к изображению

    lpr = LicensePlateRecognition()
    result = await lpr.get_plates(path_to_image)
    print(f"Результат: {result}")

    if result:
        if is_valid_plate(result):
            print("Номер валиден")
        else:
            print("Внимание: номер не прошёл валидацию")


if __name__ == "__main__":
    asyncio.run(main())
