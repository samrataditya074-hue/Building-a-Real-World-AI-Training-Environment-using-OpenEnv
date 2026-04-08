# Hybrid Customer Support & Business Operations

A real-world OpenEnv simulation environment mapping customer support logic and business operations impact.

## Features
- **Dynamic Tickets**: Generates `login_issue`, `payment_failure`, `refund_request`, and `delivery_delay` tickets with emotional tones (`angry`, `polite`, `confused`).
- **Action Space**: `categorize`, `reply`, `escalate`.
- **Hybrid Metrics**: Measures not just task completion but *Customer Satisfaction* and *Business Costs*. 

## Core Mechanics
- The agent must use multi-step intelligence to classify text, frame helpful/polite replies, and prudently escalate.
- Rewards incentivize both **Support Quality** (+0.4 for helpful response) and **Business Impact** (+0.2 high satisfaction, -0.3 unnecessary escalation cost).

## Running
Local API and UI:
```bash
uvicorn customer_support_env.app:app --host 0.0.0.0 --port 7860
```

Inference verification:
```bash
export OPENAI_API_KEY=your_key
python customer_support_env/inference.py
```
