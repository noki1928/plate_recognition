from paddleocr import TextRecognition

import logging


logger = logging.getLogger(__name__)


class Recognition:
    def __init__(self, model_name="PP-OCRv5_mobile_rec", model_dir="models/recognition"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model = TextRecognition(model_name=self.model_name, model_dir=self.model_dir)

    def get_plates(self, list_of_results):
        text = []
        
        if list_of_results is None or list_of_results == []:
            logger.warning("Пустой список изображений для распознавания")
            return ""

        for img in list_of_results:
            try:
                output = self.model.predict(input=img, batch_size=1)
                
                if output and len(output) > 0 and 'rec_text' in output[0]:
                    rec_text = output[0]['rec_text']
                    if rec_text:
                        text.append(rec_text)
                    else:
                        logger.warning("OCR вернул пустой текст")
                        text.append("UNKNOWN")
                else:
                    logger.warning("OCR не вернул результат")
                    text.append("UNKNOWN")
                    
            except Exception as e:
                logger.error(f"Ошибка распознавания текста: {e}")
                text.append("UNKNOWN")

        return ' '.join(text) if text else ""
