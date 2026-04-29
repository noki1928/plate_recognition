# Plate Recognition API

Сервис на FastAPI для распознавания автомобильных номеров по изображению.

## Что делает сервис

- Поднимает HTTP API для распознавания номера с картинки.
- При старте скачивает модель детекции `detection-v2.pt` из Hugging Face.
- Возвращает результат распознавания в JSON.

## Требования

- Python 3.12 (для запуска без Docker).

## Запуск с Docker

Сборка и первый запуск:

```bash
docker compose up --build
```

Остановка:

```bash
docker compose down
```

## Запуск без Docker

1. Создайте и активируйте виртуальное окружение:

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Запустите приложение:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

## Использование API

### POST `/recognize/`

Отправьте изображение в поле `file` (multipart/form-data).

Пример:

```bash
curl -X POST "http://localhost:8000/recognize/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

### GET `/health`

Возвращает состояние сервиса:

```json
{"status":"healthy"}
```