# ── Stage 1: build ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (models/ and data/ are expected to be present)
COPY . .

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Expose port expected by IBM Cloud Code Engine
EXPOSE 8080

# watsonx credentials — set via Code Engine environment variables, NOT baked in
# ENV WATSONX_URL=""
# ENV WATSONX_APIKEY=""
# ENV WATSONX_PROJECT_ID=""

CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8080", "--server.address=0.0.0.0"]
