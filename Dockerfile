FROM python:3.10-slim

# ── Build-time argument for OpenAI API key (optional) ─────────────────────────
# Pass it during build with: docker build --build-arg OPENAI_API_KEY=sk-... .
# Or at runtime with:        docker run -e OPENAI_API_KEY=sk-... ...
ARG OPENAI_API_KEY=""
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Set up working directory
WORKDIR /app

# Install system dependencies (needed for some numpy/scipy builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies as root first for docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (standard for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy project files and ensure correct ownership
COPY --chown=user . /app

# Hugging Face Spaces defaults to routing port 7860
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will mark container as unhealthy if the dashboard doesn't respond.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Default command: Start the Gradio + OpenEnv FastAPI dashboard ─────────────
# Alternative — run baseline only:
#   docker run -e OPENAI_API_KEY=... ceo-sim python baseline_inference.py --seed 42
ENV ENABLE_WEB_INTERFACE=false
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
