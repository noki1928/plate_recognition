import torch
from torchvision import transforms
import cv2
import numpy as np

import logging

from models.regression.plate_correction import PlateCorrectionResNet


logger = logging.getLogger(__name__)


class Correction:
    def __init__(self, path_to_model=r'models/regression/regression.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PlateCorrectionResNet()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f=path_to_model, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_correction(self, list_of_rois):

        if list_of_rois is None or list_of_rois == []:
            logger.warning("Пустой список ROI для коррекции")
            return []

        list_of_results = []

        for idx, img in enumerate(list_of_rois):
            try:
                img_to_model = self.transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    points = self.model(img_to_model)
                    points = points.cpu().numpy()[0]

                    for i in range(4):
                        points[i][0] = points[i][0] * img.shape[1]
                        points[i][1] = points[i][1] * img.shape[0]

                    matrix = cv2.getPerspectiveTransform(points.astype("float32"), np.array([[0, 0], [180, 0], [180, 40], [0, 40]]).astype("float32"))
                    result = cv2.warpPerspective(img, matrix, (180, 40))

                    list_of_results.append(result)
                    
            except Exception as e:
                logger.error(f"Ошибка коррекции изображения {idx}: {e}")
                list_of_results.append(img)

        return list_of_results
