"""Модуль распознавания текста автомобильных номеров."""

import logging
from typing import List, Optional

import numpy as np
from paddleocr import TextRecognition

logger = logging.getLogger(__name__)


class Recognition:
    """Класс для распознавания текста на изображениях номеров."""

    def __init__(
        self,
        model_name: str = "PP-OCRv5_mobile_rec",
        model_dir: str = "models/recognition",
    ) -> None:
        """
        Инициализирует модель распознавания текста.

        Args:
            model_name: Имя модели OCR.
            model_dir: Путь к директории с моделью.
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.model: Optional[TextRecognition] = None
        self._load_model()

    def _load_model(self) -> None:
        """Загружает модель OCR."""
        try:
            self.model = TextRecognition(
                model_name=self.model_name,
                model_dir=self.model_dir,
            )
            logger.info(f"Модель распознавания загружена: {self.model_name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели распознавания: {e}")
            self.model = None

    async def get_plates(self, list_of_results: List[np.ndarray]) -> str:
        """
        Распознает текст на изображениях номеров.

        Args:
            list_of_results: Список изображений ROI номеров.

        Returns:
            Распознанный текст (номера), разделенный пробелами.
        """
        if self.model is None:
            logger.error("Модель распознавания не инициализирована")
            return ""

        if not list_of_results:
            logger.warning("Пустой список изображений для распознавания")
            return ""

        text: List[str] = []

        for img in list_of_results:
            try:
                output = self.model.predict(input=img, batch_size=1)

                if output and len(output) > 0 and "rec_text" in output[0]:
                    rec_text = output[0]["rec_text"]
                    if rec_text and (len(rec_text) == 8 or len(rec_text) == 9):
                        text.append(rec_text)
                    else:
                        logger.warning("OCR вернул пустой текст или номер неверного формата")
                        # text.append("UNKNOWN")
                else:
                    logger.warning("OCR не вернул результат")
                    text.append("UNKNOWN")

            except Exception as e:
                logger.error(f"Ошибка распознавания текста: {e}")
                text.append("UNKNOWN")

        return " ".join(text) if text else ""
