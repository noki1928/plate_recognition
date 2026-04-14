from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image
import io
import logging

from plate_recognition import LicensePlateRecognition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск сервиса распознавания номеров...")
    yield
    logger.info("Сервис остановлен")


app = FastAPI(title="Распознавание автомобильных номеров", lifespan=lifespan)

model_path = hf_hub_download(
    repo_id="noki1928/russian-plates-models",
    filename="detection-v2.pt",
    local_dir="plate_recognition/models/detection"
)

lpr = LicensePlateRecognition(
    path_to_det="plate_recognition/models/detection/detection-v2.pt",
    path_to_rec="plate_recognition/models/recognition", 
    path_to_rot="plate_recognition/models/rotation/rotation.pth"
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/recognize/")
async def recognize_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Файл должен быть изображением")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = await lpr.get_plates_with_validation(image)

        logger.info(f"Распознан номер: {result.get('plates')}")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Ошибка обработки: {e}", exc_info=True)
        raise HTTPException(500, detail="Внутренняя ошибка сервера")