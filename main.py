import logging
import re

from detection import Detection
from recognition import Recognition
from regression import Correction


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PLATE_PATTERN = re.compile(
    r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$'
    r'|'
    r'^[АВЕКМНОРСТУХ]\d{4}[АВЕКМНОРСТУХ]{2}\d{2,3}$'
)


def is_valid_plate(text: str) -> bool:
    cleaned = text.upper().replace(" ", "")
    return bool(PLATE_PATTERN.match(cleaned))


class LicensePlateRecognition:
    def __init__(self, path_to_det=r'models\detection.pt', path_to_reg=r'models\regression\regression.pth', path_to_rec=r'models\recognition'):
        try:
            self.det = Detection(path_to_det)
            logger.info("Модель детекции загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели детекции: {e}")
            self.det = None

        try:
            self.reg = Correction(path_to_reg)
            logger.info("Модель коррекции загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели коррекции: {e}")
            self.reg = None

        try:
            self.rec = Recognition(model_dir=path_to_rec)
            logger.info("Модель распознавания загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели распознавания: {e}")
            self.rec = None

    def get_plates(self, path_to_image):
        if self.det is None:
            logger.error("Модель детекции не инициализирована")
            return ""

        try:
            img_rois = self.det.get_boxes(path_to_image)
            if not img_rois:
                logger.warning("Номера не найдены")
                return ""
            logger.info(f"Найдено номеров: {len(img_rois)}")
        except Exception as e:
            logger.error(f"Ошибка на этапе детекции: {e}")
            return ""

        text_without_correction = ""
        if self.rec is not None:
            try:
                text_without_correction = self.rec.get_plates(img_rois)
                logger.info(f"Распознанный текст (без коррекции): {text_without_correction}")
            except Exception as e:
                logger.error(f"Ошибка распознавания без коррекции: {e}")
        else:
            logger.error("Модель распознавания не инициализирована")

        text_with_correction = ""
        if self.reg is not None:
            try:
                img_rois_corrected = self.reg.get_correction(img_rois)
                if img_rois_corrected and self.rec is not None:
                    text_with_correction = self.rec.get_plates(img_rois_corrected)
                    logger.info(f"Распознанный текст (с коррекцией): {text_with_correction}")
                elif not img_rois_corrected:
                    logger.warning("Коррекция не вернула результатов")
            except Exception as e:
                logger.error(f"Ошибка на этапе коррекции: {e}")
        else:
            logger.warning("Модель коррекции отключена")

        result = self._select_valid_plate(text_without_correction, text_with_correction)
        if result:
            logger.info(f"Итоговый результат: {result}")
        return result

    def _select_valid_plate(self, text_without_correction: str, text_with_correction: str) -> str:
        valid_without = text_without_correction and is_valid_plate(text_without_correction)
        valid_with = text_with_correction and is_valid_plate(text_with_correction)

        if valid_without and valid_with and text_without_correction == text_with_correction:
            return text_without_correction

        if valid_without and valid_with:
            logger.info("Оба варианта валидны, но разные — выбран вариант с коррекцией")
            return text_with_correction

        if valid_with:
            logger.info("Валиден только вариант с коррекцией")
            return text_with_correction

        if valid_without:
            logger.info("Валиден только вариант без коррекции")
            return text_without_correction

        if text_with_correction:
            logger.warning("Ни один вариант не валиден, возвращаем с коррекцией")
            return text_with_correction

        if text_without_correction:
            logger.warning("Ни один вариант не валиден, возвращаем без коррекции")
            return text_without_correction

        logger.warning("Оба варианта пусты")
        return ""


if __name__ == "__main__":
    path_to_image = r"" # путь к изображению

    lpr = LicensePlateRecognition()
    result = lpr.get_plates(path_to_image)
    print(f"Результат: {result}")

    if result:
        if is_valid_plate(result):
            print("Номер валиден")
        else:
            print("Внимание: номер не прошёл валидацию")
