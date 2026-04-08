# 🏢 Autonomous CEO AI Simulator

> **In plain English:** An AI agent that runs a virtual startup. Every quarter it decides whether to raise prices, hire staff, invest in technology, or respond to crises — exactly like a real executive. Read its thoughts live or race against it in CEO Clash mode.

### 🚀 One-Command Quick Demo (No UI, Plain English)
```bash
pip install -r requirements.txt
python demo_nontech.py --quarters 10
```

### 🧠 Run the Deterministic Evaluator (Hackathon Baselines)
The baseline script natively evaluates all 3 tasks and supports multiple backends. Output automatically records the backend, model/version, token estimate, and seed.
```bash
# 1. Zero-cost reproducing evaluation (Heuristic Fallback)
python baseline_inference.py --seed 42

# 2. OpenAI GPT-4o-mini Evaluation
OPENAI_API_KEY="sk-..." python baseline_inference.py --seed 42 --openai-model gpt-4o-mini

# 3. Grok / Hugging Face Compatible (via OpenEnv backend routing if enabled)
OPENAI_API_KEY="custom-api-key" OPENAI_BASE_URL="..." python baseline_inference.py --openai-model custom-endpoint
```
