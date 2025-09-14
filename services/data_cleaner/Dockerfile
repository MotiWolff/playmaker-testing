FROM python:3.11-slim

WORKDIR /app

COPY services/data_cleaner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY services/data_cleaner/ /app/

CMD ["python", "main.py"]
