# Russian License Plate Recognition

Распознавание российских автомобильных номеров (ГОСТ Р 50577-2018 + старые форматы).

## Установка
Создать директорию проекта и в ней:
```bash
git clone https://github.com/noki1928/plate_recognition.git

python -m venv venv
source venv/bin/activate    # Linux
venv\Scripts\activate       # Windows

pip install -r plate_recognition/requirements.txt
```

## Первый запуск

Создать файл first_run.py в текущей директории

``` Python
import asyncio
from huggingface_hub import hf_hub_download
from plate_recognition import LicensePlateRecognition

model_path = hf_hub_download(
    repo_id="noki1928/russian-plates-models",
    filename="detection-v2.pt",
    local_dir="plate_recognition/models/detection"
)

path_to_image = r"plate_recognition\img\image.jpg"

async def main() -> None:
    lpr = LicensePlateRecognition(path_to_det=model_path, path_to_rec=r"plate_recognition\models\recognition")
    result = await lpr.get_plates(path_to_image)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```
Затем можно запускать

``` bash
python first_run.py
```

После этого будет установлена необходимая модель детекции с Hugging Face, а также показан результат работы распознавания текста тестового изображения.
