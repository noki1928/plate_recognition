from ultralytics import YOLO
import cv2

import logging

logger = logging.getLogger(__name__)


class Detection:
    def __init__(self, model_path=r'models\detection.pt'):
        self.model = YOLO(model_path)

    def get_boxes(self, path_to_image):
        list_of_rois = []

        try:
            img = cv2.imread(path_to_image)
            if img is None:
                logger.error(f"Не удалось загрузить изображение: {path_to_image}")
                return list_of_rois

            results = self.model.predict(path_to_image, conf=0.5)

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    logger.warning("Номера не найдены на изображении")
                    continue

                try:
                    bbox = result.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)

                    x1 = x1 - 10 if x1 - 10 > 0 else 0
                    x2 = x2 + 10 if x2 + 10 < img.shape[1] else img.shape[1]
                    y1 = y1 - 10 if y1 - 10 > 0 else 0
                    y2 = y2 + 10 if y2 + 10 < img.shape[0] else img.shape[0]

                    img_roi = img[y1:y2, x1:x2]
                    list_of_rois.append(img_roi)
                except Exception as e:
                    logger.error(f"Ошибка обработки bounding box: {e}")
                    continue

        except Exception as e:
            logger.error(f"Ошибка детекции: {e}")

        return list_of_rois






