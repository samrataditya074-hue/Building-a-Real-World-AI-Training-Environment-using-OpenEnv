FROM python:3.10-slim

WORKDIR /app

# Copy requirement and pyproject configs
COPY customer_support_env/requirements.txt requirements.txt
COPY customer_support_env/pyproject.toml pyproject.toml

# Install dependencies (ignoring heavy RL/plot libs)
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY customer_support_env /app/customer_support_env

# Expose gradio port
EXPOSE 7860

# Command to run API & UI
CMD ["python", "-m", "customer_support_env.app"]
