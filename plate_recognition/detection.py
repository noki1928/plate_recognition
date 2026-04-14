"""Модуль детекции автомобильных номеров на изображениях."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import math
import numpy as np
import torch
import torchvision
from PIL import Image as PILImage
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
        rotation_model_path: str = r"models\rotation\rotation.pth",
    ) -> None:
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self._load_model()
        
        self.rotation_model = torchvision.models.efficientnet_v2_s()
        self.rotation_model.classifier[1] = torch.nn.Linear(self.rotation_model.classifier[1].in_features, 4)
        state_dict = torch.load(rotation_model_path, map_location=torch.device('cpu'))
        self.rotation_model.load_state_dict(state_dict)
        self.rotation_model.eval()
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Модель детекции загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели детекции: {e}")
            self.model = None

    async def get_rois(
        self,
        source: Union[str, Path, np.ndarray, PILImage.Image],
        conf: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Извлекает ROI номеров.

        source может быть:
            - str / Path → путь к файлу
            - np.ndarray → изображение в BGR
            - PIL.Image → изображение в RGB
        """
        list_of_rois: List[np.ndarray] = []

        if self.model is None:
            logger.error("Модель детекции не инициализирована")
            return list_of_rois

        try:
            # 1. Определяем, что пришло, и готовим img (BGR) + source для YOLO
            if isinstance(source, (str, Path)):
                img = cv2.imread(str(source))
                if img is None:
                    logger.error(f"Не удалось загрузить изображение: {source}")
                    return list_of_rois

            elif isinstance(source, PILImage.Image):
                img = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

            elif isinstance(source, np.ndarray):
                img = source.copy()          # предполагаем BGR

            else:
                raise ValueError(f"Неподдерживаемый тип source: {type(source)}")

            # 2. Применяем поворот через модель rotation
            rotate_input = img.copy()
            output = self.rotation_model(self.transform(rotate_input).unsqueeze(0))
            angle_idx = output.argmax(dim=1).item()
            predicted_angle = [0, 90, 180, 270][angle_idx]

            if predicted_angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif predicted_angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif predicted_angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 3. Запускаем YOLO (теперь можно передавать numpy/PIL напрямую)
            results = self.model.predict(img, conf=conf, verbose=False)

            for result in results:
                if result.keypoints is None or len(result.keypoints) == 0:
                    logger.warning("Ключевые точки не найдены")
                    continue

                try:
                    for kp in result.keypoints.xy.cpu().numpy():
                        points = kp.astype(int)
                        x_coords = points[:, 0]
                        y_coords = points[:, 1]

                        sorted_points = sort_points(x_coords, y_coords)
                        side1, side2 = calculate_max_side(sorted_points)

                        if side2 / side1 > 2:          # одна строка
                            dst = np.array([[0, 0], [180, 0], [180, 40], [0, 40]], dtype="float32")
                            roi_size = (180, 40)
                        else:                          # две строки
                            dst = np.array([[0, 0], [90, 0], [90, 80], [0, 80]], dtype="float32")
                            roi_size = (90, 80)

                        matrix = cv2.getPerspectiveTransform(
                            sorted_points.astype("float32"), dst
                        )
                        roi = cv2.warpPerspective(img, matrix, roi_size)

                        if roi is not None:
                            if roi_size == (90, 80):   # превращаем в 2 строки
                                roi_2line = np.zeros((40, 180, 3), dtype=np.uint8)
                                roi_2line[:, :90] = roi[0:40, :]
                                roi_2line[:, 90:] = roi[40:80, :]
                                list_of_rois.append(roi_2line)
                            else:
                                list_of_rois.append(roi)

                except Exception as e:
                    logger.error(f"Ошибка обработки keypoints: {e}")
                    continue

        except Exception as e:
            logger.error(f"Ошибка детекции: {e}")

        return list_of_rois