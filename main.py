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

        if self.reg is not None:
            try:
                img_rois = self.reg.get_correction(img_rois)
                if not img_rois:
                    logger.warning("Коррекция не вернула результатов")
            except Exception as e:
                logger.error(f"Ошибка на этапе коррекции: {e}")
        else:
            logger.warning("Модель коррекции отключена")

        if self.rec is not None:
            try:
                text = self.rec.get_plates(img_rois)
                if text:
                    logger.info(f"Распознанный текст: {text}")
                    return text
                else:
                    logger.warning("Распознавание не вернуло текст")
                    return ""
            except Exception as e:
                logger.error(f"Ошибка на этапе распознавания: {e}")
                return ""
        else:
            logger.error("Модель распознавания не инициализирована")
            return ""


if __name__ == "__main__":
    path_to_image = r'C:\progs\datasets\znaki\yolo_train\train\images\351.jpg'

    lpr = LicensePlateRecognition()
    result = lpr.get_plates(path_to_image)
    print(f"Результат: {result}")

    if result:
        if is_valid_plate(result):
            print("Номер валиден")
        else:
            print("Внимание: номер не прошёл валидацию")
