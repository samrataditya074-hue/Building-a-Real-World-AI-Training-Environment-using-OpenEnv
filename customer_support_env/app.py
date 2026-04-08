"""
customer_support_env/app.py — FastAPI + Gradio server (OpenEnv-compliant).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import uvicorn
from fastapi import FastAPI

from openenv.core.env_server import create_app
from customer_support_env.models import Action, Observation, State
from customer_support_env.environment import SupportEnvironment

_demo_env = SupportEnvironment()
app = create_app(SupportEnvironment, Action, Observation, State)

def _urgency_badge(urgency: str) -> str:
    colors = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}
    c = colors.get(urgency, "#94a3b8")
    return f"<span style='background:{c}; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:700;'>{urgency.upper()}</span>"

def _format_history(history: list) -> str:
    if not history:
        return "<p style='color:#64748b; font-style:italic;'>No messages yet.</p>"
    lines = []
    for msg in history:
        if msg.startswith("[Customer]"):
            lines.append(f"<div style='background:#1e3a5f; border-left:4px solid #3b82f6; padding:10px; margin:8px 0; border-radius:6px;'>👤 {msg[10:]}</div>")
        elif "[CATEGORIZE]" in msg:
            lines.append(f"<div style='background:#14532d; border-left:4px solid #22c55e; padding:10px; margin:8px 0; border-radius:6px;'>🏷️ {msg}</div>")
        elif "[REPLY]" in msg:
            lines.append(f"<div style='background:#312e81; border-left:4px solid #818cf8; padding:10px; margin:8px 0; border-radius:6px;'>💬 {msg}</div>")
        elif "[ESCALATE]" in msg:
            lines.append(f"<div style='background:#7f1d1d; border-left:4px solid #ef4444; padding:10px; margin:8px 0; border-radius:6px;'>🚨 {msg}</div>")
        else:
            lines.append(f"<div>{msg}</div>")
    return "".join(lines)

def ui_reset(seed_val: str):
    seed = int(seed_val) if seed_val.strip().isdigit() else None
    obs = _demo_env.reset(seed=seed)
    s = _demo_env.typed_state()

    history_html = _format_history(s.conversation_history)
    status_md = (
        f"**Step:** 0 / {_demo_env.max_steps}  |  **Resolved:** ❌  |  **Reward:** —\n\n"
        f"**Sat Score:** {s.satisfaction_score:.2f} | **Cost Spent:** ${s.cost_spent:.2f} | **Tickets Resolved:** {s.tickets_resolved}"
    )
    return f"<h3>{s.ticket_id} {_urgency_badge(s.urgency)}</h3><p>{s.customer_message}</p>", history_html, status_md, "", ""

def ui_step(action_type: str, content: str):
    if not action_type or not content.strip():
        return _format_history(_demo_env.typed_state().conversation_history), "⚠️ Fill action and content", ""
    
    try:
        action = Action(action_type=action_type, content=content.strip())
        obs = _demo_env.step(action)
    except Exception as exc:
        return _format_history(_demo_env.typed_state().conversation_history), f"❌ Invalid action: {exc}", ""

    s = _demo_env.typed_state()
    history_html = _format_history(s.conversation_history)
    
    bd_str = " | ".join(f"{k}: {v:+.2f}" for k, v in obs.info.get("reward_breakdown", {}).items())
    done_msg = "\n\n🏁 **RESOLVED/DONE!** Click Reset." if obs.done else ""

    status_md = (
        f"**Step:** {s.step_count} / {_demo_env.max_steps}  |  **Resolved:** {'✅' if s.resolved else '❌'}  |  **Reward:** `{obs.reward:.3f}`\n\n"
        f"**Sat Score:** {s.satisfaction_score:.2f} | **Cost Spent:** ${s.cost_spent:.2f} | **Tickets Resolved:** {s.tickets_resolved}\n\n"
        f"**Breakdown:** {bd_str}"
        f"{done_msg}"
    )
    return history_html, status_md, ""

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🎫 Hybrid Support + Business Environment")
    with gr.Row():
        with gr.Column(scale=2):
            ticket_display = gr.HTML("Load a ticket by clicking Reset.")
            history_display = gr.HTML("")
        with gr.Column(scale=1):
            action_type_dd = gr.Dropdown(choices=["categorize", "reply", "escalate"], value="categorize", label="Action")
            content_in = gr.Textbox(label="Content", lines=3)
            step_btn = gr.Button("Execute Action")
            status_md = gr.Markdown("Status Pending...")
            seed_in = gr.Textbox(label="Seed", value="42")
            reset_btn = gr.Button("Reset Episode")

    reset_btn.click(ui_reset, inputs=[seed_in], outputs=[ticket_display, history_display, status_md, content_in, action_type_dd])
    step_btn.click(ui_step, inputs=[action_type_dd, content_in], outputs=[history_display, status_md, content_in])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
