# Dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Install supervisord + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install all Python requirements
COPY requirements.txt .
COPY model_api/requirements.txt model_api_requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r model_api_requirements.txt \
    && rm model_api_requirements.txt

# Copy code
COPY app.py .
COPY tools ./tools
COPY model_api/resnet_api.py ./model_api/

# Expose ports
EXPOSE 8501  # Streamlit
EXPOSE 8000  # FastAPI

# Create supervisord config
RUN echo '[supervisord]\n\
nodaemon=true\n\
user=root\n\
\n\
[program:streamlit]\n\
command=streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stderr_logfile=/dev/stderr\n\
\n\
[program:fastapi]\n\
command=uvicorn model_api.resnet_api:app --host 0.0.0.0 --port 8000\n\
directory=/app\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stderr_logfile=/dev/stderr\n\
environment=RUN_API="true"' > /etc/supervisor/conf.d/supervisord.conf

# Final command
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
