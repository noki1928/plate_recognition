"""Модуль детекции автомобильных номеров на изображениях."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import math
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def calculate_max_side(points: np.ndarray) -> Tuple[float, float]:
    """
    Вычисляет максимальные длины сторон прямоугольника по точкам.

    Args:
        points: Массив точек размером (4, 2) в формате [[x0, y0], [x1, y1], [x2, y2], [x3, y3]].

    Returns:
        Кортеж (side1, side2) с максимальными длинами противоположных сторон.
    """
    side11 = math.sqrt(
        (points[0][0] - points[3][0]) ** 2
        + (points[0][1] - points[3][1]) ** 2
    )
    side12 = math.sqrt(
        (points[1][0] - points[2][0]) ** 2
        + (points[1][1] - points[2][1]) ** 2
    )

    side21 = math.sqrt(
        (points[0][0] - points[1][0]) ** 2
        + (points[0][1] - points[1][1]) ** 2
    )
    side22 = math.sqrt(
        (points[3][0] - points[2][0]) ** 2
        + (points[3][1] - points[2][1]) ** 2
    )

    return max(side11, side12), max(side21, side22)


def sort_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Сортирует 4 точки в порядке: верхний-левый, верхний-правый, нижний-правый, нижний-левый.

    Args:
        x: Массив x-координат точек.
        y: Массив y-координат точек.

    Returns:
        Отсортированный массив точек размером (4, 2).
    """
    c_x = (x[0] + x[1] + x[2] + x[3]) / 4
    c_y = (y[0] + y[1] + y[2] + y[3]) / 4

    points = list(zip(x, y))
    points.sort(key=lambda p: math.atan2(p[1] - c_y, p[0] - c_x))

    first_points = [point for point in points if point[0] < c_x]

    if not first_points:
        first_point = min(points, key=lambda p: p[1])
    else:
        first_point = min(first_points, key=lambda p: p[1])

    i = points.index(first_point)
    points = points[i:] + points[:i]

    return np.array(points)


class Detection:
    """Класс для детекции автомобильных номеров на изображениях."""

    def __init__(
        self,
        model_path: str = r"models\detection\detection-v2.pt",
    ) -> None:
        """
        Инициализирует модель детекции.

        Args:
            model_path: Путь к файлу модели YOLO.
        """
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Загружает модель YOLO."""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Модель детекции загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели детекции: {e}")
            self.model = None

    async def get_rois(
        self,
        path_to_image: str | Path,
        conf: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Извлекает ROI (Region of Interest) автомобильных номеров из изображения.

        Args:
            path_to_image: Путь к изображению.
            conf: Порог уверенности детекции (0.0 - 1.0).

        Returns:
            Список изображений ROI номеров.
        """
        list_of_rois: List[np.ndarray] = []

        if self.model is None:
            logger.error("Модель детекции не инициализирована")
            return list_of_rois

        path_to_image = Path(path_to_image)

        try:
            img = cv2.imread(str(path_to_image))
            if img is None:
                logger.error(f"Не удалось загрузить изображение: {path_to_image}")
                return list_of_rois

            results = self.model.predict(str(path_to_image), conf=conf)

            for result in results:
                if result.keypoints is None or len(result.keypoints) == 0:
                    logger.warning("Номера не найдены на изображении")
                    continue

                try:
                    for kp in result.keypoints.xy.cpu().numpy():
                        points = kp.astype(int)

                        x_coords = points[:, 0]
                        y_coords = points[:, 1]

                        sorted_points = sort_points(x_coords, y_coords)
                        side1, side2 = calculate_max_side(sorted_points)

                        matrix: Optional[np.ndarray] = None
                        roi: Optional[np.ndarray] = None

                        if side2 / side1 > 2:
                            matrix = cv2.getPerspectiveTransform(
                                sorted_points.astype("float32"),
                                np.array(
                                    [[0, 0], [180, 0], [180, 40], [0, 40]]
                                ).astype("float32"),
                            )
                            roi = cv2.warpPerspective(img, matrix, (180, 40))
                            if roi is not None:
                                list_of_rois.append(roi)
                        else:
                            matrix = cv2.getPerspectiveTransform(
                                sorted_points.astype("float32"),
                                np.array(
                                    [[0, 0], [90, 0], [90, 80], [0, 80]]
                                ).astype("float32"),
                            )
                            roi = cv2.warpPerspective(img, matrix, (90, 80))

                            if roi is not None:
                                roi_2line = np.zeros((40, 180, 3), dtype=np.uint8)
                                roi_2line[:, :90] = roi[0:40, :]
                                roi_2line[:, 90:] = roi[40:80, :]
                                list_of_rois.append(roi_2line)

                except Exception as e:
                    logger.error(f"Ошибка обработки keypoints: {e}")
                    continue

        except Exception as e:
            logger.error(f"Ошибка детекции: {e}")

        return list_of_rois
