# LG헬로비전 고객 분석 대시보드 (Streamlit)
# build:  docker build -t lghv-dashboard .
# run:    docker run -p 8501:8501 lghv-dashboard
FROM python:3.11-slim

WORKDIR /app

# 헬스체크용 curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 대시보드 전용 경량 의존성 (streamlit · plotly · pandas · numpy)
COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 앱 엔트리 + 데모 데이터 + 테마
COPY streamlit_app.py ./streamlit_app.py
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
