FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for xgboost / scipy / shap
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY core/ ./core/
COPY web/ ./web/
COPY app/ ./app/
COPY examples/ ./examples/
COPY prompts/ ./prompts/

EXPOSE 8000 8501

# Default: FastAPI (also serves the web UI at /).
# To run Streamlit instead:
#   docker run -p 8501:8501 IMAGE streamlit run app/streamlit_app.py --server.address 0.0.0.0
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
