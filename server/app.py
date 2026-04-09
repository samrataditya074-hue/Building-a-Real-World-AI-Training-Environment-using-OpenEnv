import gradio as gr
import numpy as np
import plotly.graph_objects as go
import time
from fastapi import FastAPI
import uvicorn
import pandas as pd
import io
import csv

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server import create_app
from models import Action, Observation, State
from server.environment import CEOEnvironment
from agent.business_agent import CorporateAgent
from graders import GRADERS

# Add LLM Support (via Groq)
try:
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    _groq_key = os.getenv("GROQ_API_KEY", "")
    has_openai = bool(_groq_key)
    openai_client = OpenAI(api_key=_groq_key, base_url="https://api.groq.com/openai/v1") if has_openai else None
except ImportError:
    has_openai = False
    openai_client = None

# Create the OpenEnv FastAPI Server instance
env = CEOEnvironment()
app = create_app(CEOEnvironment, Action, Observation, State)

# ========== GLOBAL DATA & HELPERS ==========
leaderboard_data = []

def export_to_csv(history):
    """Converts metrics history to a downloadable CSV string."""
    if not history:
        return None
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
    return output.getvalue()

# Metadata for Grader & UI reporting
TASK_METADATA = {
    "easy_revenue_target": {"name": "Revenue Growth Benchmark", "difficulty": "Easy"},
    "medium_budget_balance": {"name": "Budget Scaling & Workforce Management", "difficulty": "Medium"},
    "hard_strategic_growth": {"name": "Long-Term Valuation Optimization", "difficulty": "Hard"},
}

# ========== GRADIO UI COMPONENTS ==========

def create_3d_landscape(dept_scores: dict):
    """Creates animated 3D bar chart of department performance."""
    depts = list(dept_scores.keys())
    scores = list(dept_scores.values())
    colors = ['#27ae60', '#f1c40f', '#3498db', '#9b59b6', '#e67e22']

    fig = go.Figure()
    for i, (dept, score, color) in enumerate(zip(depts, scores, colors)):
        x0, x1 = i * 1.5, i * 1.5 + 1
        fig.add_trace(go.Mesh3d(
            x=[x0, x1, x1, x0, x0, x1, x1, x0],
            y=[0, 0, 1, 1, 0, 0, 1, 1],
            z=[0, 0, 0, 0, score, score, score, score],
            color=color, opacity=0.85, name=dept,
            i=[0,0,4,4,0,0,2,2,0,0,1,1],
            j=[1,2,5,6,1,4,3,6,4,1,5,2],
            k=[2,3,6,7,4,5,6,7,5,5,6,6],
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='', showticklabels=False, showgrid=False),
            yaxis=dict(title='', showticklabels=False, showgrid=False),
            zaxis=dict(title='Score', range=[0, 12]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            bgcolor='rgba(17,17,17,0.9)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40),
        title=dict(text="🏢 3D Corporate Ecosystem", font=dict(color='white', size=16)),
        legend=dict(font=dict(color='white')),
        font=dict(color='white'),
    )
    return fig

def create_reward_plot(steps, pos_hist, neg_hist, action_markers):
    """Live reward analytics with action markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=pos_hist, name="Growth (+)",
        mode='lines',
        line=dict(color='#2ecc71', width=3), fill='tozeroy',
        fillcolor='rgba(46,204,113,0.15)',
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=neg_hist, name="Risk (-)",
        mode='lines',
        line=dict(color='#e74c3c', width=3), fill='tozeroy',
        fillcolor='rgba(231,76,60,0.15)',
    ))
    
    # Add action markers at notable steps
    for step_i, label in action_markers[-5:]:
        fig.add_vline(x=step_i, line_dash="dot", line_color="rgba(255,255,255,0.8)")
        fig.add_annotation(
            x=step_i, y=max(max(pos_hist) if pos_hist else 0, max(neg_hist) if neg_hist else 0) * 0.9,
            text="*", showarrow=False, font=dict(color="white", size=18)
        )

    fig.update_layout(
        title=dict(text="📊 Live Reward Tracking", font=dict(color='white', size=14)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)',
        font=dict(color='white'),
        xaxis=dict(title="Quarter", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Reward", gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(font=dict(color='white')),
        margin=dict(l=40, r=20, b=40, t=40),
    )
    return fig

def create_competitor_plot(steps, our_prices, comp_prices):
    """Plot comparing our price vs competitor price over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=our_prices, name="Our Price",
        mode='lines', line=dict(color='#3498db', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=comp_prices, name="Competitor Price",
        mode='lines', line=dict(color='#e74c3c', width=3, dash='dash')
    ))
    fig.update_layout(
        title=dict(text="📈 Pricing War Analytics", font=dict(color='white', size=14)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)',
        font=dict(color='white'),
        xaxis=dict(title="Quarter", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Price ($)", gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(font=dict(color='white'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, b=40, t=40),
    )
    return fig

def create_valuation_plot(steps, valuations):
    """Plot tracking overall business health (growth vs downfall) over time."""
    fig = go.Figure()
    
    colors = ['#2ecc71' if i > 0 and v > valuations[i-1] else '#e74c3c' for i, v in enumerate(valuations)]
    if not colors: colors = ['#2ecc71']
    
    fig.add_trace(go.Bar(
        x=steps, y=valuations, name="Valuation ($)",
        marker_color=colors
    ))
    
    # Add a trend line
    fig.add_trace(go.Scatter(
        x=steps, y=valuations, name="Trend",
        mode='lines', line=dict(color='#f1c40f', width=2, dash='dot')
    ))

    fig.update_layout(
        title=dict(text="📊 Corporate Growth vs Downfall", font=dict(color='white', size=14)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)',
        font=dict(color='white'),
        xaxis=dict(title="Quarter", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Intrinsic Valuation", gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False,
        margin=dict(l=40, r=20, b=40, t=40),
    )
    return fig

def format_metrics(s: State):
    """Format status bar with key metrics and crisis alerts."""
    crisis = []
    if s.cash_crisis: crisis.append("💸 CASH")
    if s.morale_crisis: crisis.append("😟 MORALE")
    if s.inventory_crisis: crisis.append("📦 INVENTORY")
    if s.reputation_crisis: crisis.append("📉 REPUTATION")
    crisis_str = " | ".join(crisis) if crisis else "✅ All Clear"

    crisis_color = "red" if crisis else "green"
    html_crisis = f"<span style='color: {crisis_color}; font-weight: bold;'>{crisis_str}</span>"

    return (
        f"### Q{s.quarter} | Cash: ${s.cash:,.0f} | Profit: ${s.profit:,.0f} | "
        f"Revenue: ${s.revenue:,.0f}\n"
        f"**Employees:** {s.total_employees} | **Morale:** {s.employee_morale:.0f}% | "
        f"**Satisfaction:** {s.customer_satisfaction:.0f}% | **R&D:** {s.rd_progress:.0f}%\n\n"
        f"**Crises:** {html_crisis}"
    )

# --- NEW: Utility function for safe RL output scaling ---
def scale_positive(x: float) -> float:
    """Safely map RL outputs from [-1.0, 1.0] to [0.0, 1.0] bounds."""
    return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))

# --- NEW: Utility function for LLM Narration (WOW Factor) ---
def get_llm_thought(s: State, base_thought: str) -> str:
    if not has_openai or not openai_client:
        return base_thought
    
    prompt = f"You are the internal monologue of a highly intelligent AI CEO. \n" \
             f"Current company metrics: Cash=${s.cash:,.0f}, Morale={s.employee_morale:.0f}%, Quarter={s.quarter}.\n" \
             f"Default hardcoded logic: {base_thought}\n" \
             f"Task: Rewrite this logic into one dramatic, impressive, and strategic sentence as if you were presenting to the board."
    try:
        res = openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.7
        )
        return "✨ [Live LLM Insight] " + res.choices[0].message.content.strip()
    except Exception:
        return base_thought

def format_actions(actions):
    """Format action list as readable bullets."""
    if not actions:
        return "No actions this quarter."
    return "\n".join([f"• {a}" for a in actions])

def stream_simulation(task_id, mode, speed="normal", user_p=0, user_m=0, user_h=0, user_r=0,
                       user_s=0, user_t=0, user_c=0, user_b=0):
    """Main simulation generator with speed control and headline support."""
    agent = CorporateAgent()
    obs = env.reset(task_id=task_id)
    delay = 0.15 if speed == "normal" else 0.01

    pos_hist, neg_hist, steps = [], [], []
    our_prices, comp_prices = [], []
    valuations = []
    action_markers = []

    for i in range(100):
        # Update delay dynamically if changed
        current_delay = 0.15 if speed == "normal" else 0.01
        
        if mode == "auto":
            action_np = agent.compute_action(obs.to_array())
            action = Action(
                price_adjustment=float(action_np[0]),
                marketing_push=scale_positive(action_np[1]), # mapped [-1, 1] -> [0, 1]
                hire_fire=float(action_np[2]),
                rd_investment=scale_positive(action_np[3]),  # mapped [-1, 1] -> [0, 1]
                salary_adjustment=float(action_np[4]),
                task_allocation=float(action_np[5]),
                crisis_response=float(action_np[6]),
                budget_shift=float(action_np[7])
            )
        else:
            action = Action(
                price_adjustment=float(user_p),
                marketing_push=float(user_m),
                hire_fire=float(user_h),
                rd_investment=float(user_r),
                salary_adjustment=float(user_s),
                task_allocation=float(user_t),
                crisis_response=float(user_c),
                budget_shift=float(user_b)
            )

        step_res = env.step(action)
        s = env.typed_state()
        obs = step_res

        pos_hist.append(step_res.info["pos_reward"])
        neg_hist.append(step_res.info["neg_reward"])
        steps.append(i + 1)
        
        our_prices.append(s.competitor_price + action.price_adjustment * 8.0)
        comp_prices.append(s.competitor_price)
        valuations.append(s.get_valuation())

        for act in step_res.info["actions"]:
            if any(kw in act.lower() for kw in ["hired", "fired", "crisis", "r&d", "salary", "budget"]):
                action_markers.append((i + 1, act))

        # Optimization: In fast-forward, we skip some UI updates for performance
        if speed == "fast" and i % 5 != 0 and not step_res.done:
            continue

        plot_3d = create_3d_landscape(s.dept_scores())
        reward_fig = create_reward_plot(steps, pos_hist, neg_hist, action_markers)
        comp_fig = create_competitor_plot(steps, our_prices, comp_prices)
        val_fig = create_valuation_plot(steps, valuations)
        
        metrics = format_metrics(s)
        thought = step_res.info["thought"]
        
        # WOW FACTOR: In normal speed auto mode, generate live LLM insights for the thought cloud
        if mode == "auto" and speed == "normal" and has_openai:
            thought = get_llm_thought(s, thought)

        actions_text = format_actions(step_res.info["actions"])
        events_text = "\n".join(s.event_history[-8:]) if s.event_history else "No major events yet."
        roster_data = s.get_roster()
        
        # New Hackathon Features
        headline_html = f"<div style='background: #111; padding: 15px; border-left: 5px solid #f1c40f; margin-bottom: 20px;'><h2 style='color: #eee; margin: 0; font-family: serif;'>{s.headline}</h2></div>"

        yield plot_3d, reward_fig, comp_fig, val_fig, metrics, thought, actions_text, events_text, roster_data, headline_html, ""

        if step_res.done:
            # Handle leaderboard
            global leaderboard_data
            final_val = s.get_valuation()
            leaderboard_data.append([f"Q{s.quarter}", f"${final_val:,.0f}", s.headline[:30]+"..."])
            leaderboard_data = sorted(leaderboard_data, key=lambda x: float(x[1].replace('$','').replace(',','')), reverse=True)[:10]
            
            if s.cash <= 0:
                reason = "💸 BANKRUPTCY"
            elif s.customer_satisfaction <= 5:
                reason = "😡 CUSTOMER EXODUS"
            elif s.total_employees <= 2:
                reason = "👻 WORKFORCE COLLAPSE"
            else:
                reason = "🏁 MAX STEPS"
            
            # Hardened Programmatic Grader Invocation
            grader_fn = GRADERS.get(task_id, GRADERS["easy_revenue_target"])
            final_score = grader_fn(env.typed_state().metrics_history)
            
            meta = TASK_METADATA.get(task_id, {"name": "Unknown", "difficulty": "N/A"})
            report_md = f"# 🏁 MISSION COMPLETE\n" \
                        f"**Task**: {meta['name']}\n" \
                        f"**Difficulty**: {meta['difficulty']}\n" \
                        f"**Final Grader Score**: {final_score:.4f}\n" \
                        f"---\n" \
                        f"*Result: {reason}*"
            
            # Yield final state with the bold report card at the end
            yield plot_3d, reward_fig, comp_fig, val_fig, metrics, thought, actions_text, events_text, roster_data, headline_html, report_md
            break

        time.sleep(current_delay)

def run_clash(p, m, h, r, s, t, c, b):
    """AI vs Human Instant Clash race."""
    human_env = CEOEnvironment()
    ai_env = CEOEnvironment()
    ai_agent = CorporateAgent()
    
    h_obs = human_env.reset()
    a_obs = ai_env.reset()
    
    h_vals, a_vals, steps = [], [], []
    
    human_action = Action(
        price_adjustment=float(p), marketing_push=float(m), hire_fire=float(h),
        rd_investment=float(r), salary_adjustment=float(s), task_allocation=float(t),
        crisis_response=float(c), budget_shift=float(b)
    )

    for i in range(50): # 50 quarter race
        # AI compute action
        a_act_np = ai_agent.compute_action(a_obs.to_array())
        ai_action = Action(
            price_adjustment=float(a_act_np[0]), 
            marketing_push=scale_positive(a_act_np[1]), # mapped [-1, 1] -> [0, 1]
            hire_fire=float(a_act_np[2]), 
            rd_investment=scale_positive(a_act_np[3]),  # mapped [-1, 1] -> [0, 1]
            salary_adjustment=float(a_act_np[4]), task_allocation=float(a_act_np[5]),
            crisis_response=float(a_act_np[6]), budget_shift=float(a_act_np[7])
        )
        
        h_res = human_env.step(human_action)
        a_res = ai_env.step(ai_action)
        
        h_vals.append(human_env.typed_state().get_valuation())
        a_vals.append(ai_env.typed_state().get_valuation())
        steps.append(i+1)
        
        a_obs = a_res
        
        # Plotting the clash
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=h_vals, name="YOU (Human)", line=dict(color='#3498db', width=4)))
        fig.add_trace(go.Scatter(x=steps, y=a_vals, name="AI (Corporate Agent)", line=dict(color='#f1c40f', width=4, dash='dash')))
        fig.update_layout(title="⚔️ CEO CLASH: Human vs AI", paper_bgcolor='rgba(17,17,17,0.9)', font=dict(color='white'))
        
        winner = "AI is leading..." if a_vals[-1] > h_vals[-1] else "YOU are leading!"
        msg = f"### Quarter {i+1}/50 | {winner}"
        
        yield fig, msg
        if h_res.done or a_res.done: break
        time.sleep(0.05)


def reset_environment(task_id):
    """Manual reset of the environment"""
    env.reset(task_id=task_id)
    s = env.typed_state()
    # Return empty graphs and starting metrics
    plot_3d = create_3d_landscape(s.dept_scores())
    reward_fig = create_reward_plot([], [], [], [])
    comp_fig = create_competitor_plot([], [], [])
    val_fig = create_valuation_plot([], [])
    metrics = format_metrics(s)
    roster_data = s.get_roster()
    return plot_3d, reward_fig, comp_fig, val_fig, metrics, "Simulation reset.", "Waiting for actions.", "Event log cleared.", roster_data, "<div style='height: 50px;'></div>", ""

with gr.Blocks(title="Autonomous CEO AI Simulator") as demo:
    gr.Markdown("# 🏢 Autonomous CEO AI Simulator")
    gr.Markdown("Watch an AI make real CEO decisions — hiring, firing, pricing, R&D, crisis management — all in real-time. Full OpenEnv API runs alongside.")

    with gr.Tabs():
        # --- TAB 1: AUTONOMOUS SIMULATION ---
        with gr.Tab("🤖 Autonomous AI Dashboard"):
            ticker = gr.HTML("<div style='background: #111; padding: 15px; border-left: 5px solid #f1c40f; margin-bottom: 20px;'><h2 style='color: #eee; margin: 0; font-family: serif;'>Waiting for CEO founding...</h2></div>")
            mission_report = gr.Markdown("", label="Final Mission Results")
            with gr.Row():
                with gr.Column(scale=2):
                    plot_3d = gr.Plot(label="3D Corporate Ecosystem")
                with gr.Column(scale=1):
                    gr.Markdown("### 💭 AI Thought Cloud")
                    thought_box = gr.Textbox(label="CEO Reasoning (Powered by Groq LLM if GROQ_API_KEY is set)", lines=3, interactive=False)
                    gr.Markdown("### ⚡ Actions Taken")
                    actions_box = gr.Textbox(label="Micro-Narrative Updates", lines=3, interactive=False)
                    gr.Markdown("### 📰 Event Log")
                    event_box = gr.Textbox(label="Global Market Events", lines=3, interactive=False)

            with gr.Row():
                val_plot = gr.Plot(label="Growth vs Downfall")
                reward_plot = gr.Plot(label="Live Reward Analytics")
                comp_plot = gr.Plot(label="Pricing War")
                
            with gr.Row():
                roster_table = gr.Dataframe(
                    headers=["Employee ID", "Name", "Department", "Salary", "Perform", "Morale", "Tenure Qtrs"],
                    label="Live Employee Roster",
                    interactive=False
                )

            metrics_md = gr.Markdown("### 1. Select Authority Level & Mission")
            with gr.Row():
                with gr.Column():
                    task_select = gr.Dropdown(
                        choices=["easy_revenue_target", "medium_budget_balance", "hard_strategic_growth"],
                        value="easy_revenue_target",
                        label="Select Task Level (Authorization Required)"
                    )
                    mission_brief = gr.Markdown("---")
                    run_btn = gr.Button("🚀 Demo Auto-Run", variant="primary", size="lg")
                    ff_mode = gr.Checkbox(label="⚡ Fast-Forward Mode", value=False)
                with gr.Column():
                    reset_btn = gr.Button("🔄 Reset Session", size="lg")
                    export_btn = gr.Button("📤 Export to CSV", variant="secondary")
                    csv_output = gr.File(label="Download Report")

            def update_brief(task):
                briefs = {
                    "easy_revenue_target": "**Goal**: Increase revenue by 10% within 1 year (4 Quarters).\n**Success**: Efficient pricing and marketing scaling.",
                    "medium_budget_balance": "**Goal**: Scale to 25+ staff with stable departmental budgets.\n**Success**: Balanced departmental share >10%.",
                    "hard_strategic_growth": "**Goal**: Maximize valuation ($500k+) through long-term R&D.\n**Success**: High innovation and brand metrics."
                }
                return briefs.get(task, "")
            
            task_select.change(fn=update_brief, inputs=[task_select], outputs=[mission_brief])

        # --- TAB 2: MANUAL CEO MODE & AI MENTOR ---
        with gr.Tab("🎮 Manual CEO Mode"):
            gr.Markdown("### Take the CEO seat. Use AI Mentor for strategy guidance.")
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        mp = gr.Slider(-1, 1, value=0, label="💰 Pricing")
                        mm = gr.Slider(0, 1, value=0, label="📣 Marketing")
                        mh = gr.Slider(-1, 1, value=0, label="👥 Hire/Fire")
                        mr = gr.Slider(0, 1, value=0, label="🧬 R&D")
                    with gr.Row():
                        ms = gr.Slider(-1, 1, value=0, label="💵 Salary")
                        mt = gr.Slider(-1, 1, value=0, label="📋 Task Alloc")
                        mc = gr.Slider(-1, 1, value=0, label="🚨 Crisis Resp")
                        mb = gr.Slider(-1, 1, value=0, label="📊 Budget Shift")
                with gr.Column(scale=1):
                    mentor_btn = gr.Button("🧠 Ask AI Mentor", variant="secondary")
                    mentor_box = gr.Textbox(label="AI Mentor Advice", lines=4)

            m_plot3d = gr.Plot(label="3D Corporate Ecosystem")
            with gr.Row():
                m_reward = gr.Plot(label="Reward Analytics")
                m_comp = gr.Plot(label="Pricing War")
                m_val = gr.Plot(label="Growth vs Downfall")
            with gr.Row():
                with gr.Column(scale=1):
                    m_thought = gr.Textbox(label="Thought Cloud", lines=2)
                    m_actions = gr.Textbox(label="Actions", lines=2)
                with gr.Column(scale=1):
                    m_events = gr.Textbox(label="Event Log", lines=2)
                    m_roster = gr.Dataframe(headers=["Employee ID", "Name", "Department", "Salary", "Perform", "Morale", "Tenure Qtrs"])
            m_report = gr.Markdown("", label="Final Mission Results")
            
            m_metrics = gr.Markdown("### Adjust and click below")
            with gr.Row():
                m_task_select = gr.Dropdown(
                    choices=["easy_revenue_target", "medium_budget_balance", "hard_strategic_growth"],
                    value="easy_revenue_target",
                    label="Active Task Context"
                )
                manual_btn = gr.Button("⏱️ Run Quarter", variant="primary")
                m_reset_btn = gr.Button("🔄 Reset Session")

        # --- TAB 3: AI vs HUMAN CLASH ---
        with gr.Tab("⚔️ Multiplayer: CEO Clash"):
            gr.Markdown("### Set your strategy and race against the AI for 50 quarters!")
            with gr.Row():
                c_p = gr.Slider(-1, 1, value=0, label="💰 Price Strategy")
                c_m = gr.Slider(0, 1, value=0, label="📣 Market Strategy")
                c_h = gr.Slider(-1, 1, value=0, label="👥 HR Strategy")
                clash_btn = gr.Button("🚩 START CLASH INSTANT RACE", variant="primary", size="lg")
            
            clash_plot = gr.Plot(label="Live Valuation Race")
            clash_msg = gr.Markdown("### Waiting for clash...")

        # --- TAB 4: LEADERBOARD ---
        with gr.Tab("🏆 Global Leaderboard"):
            leaderboard_ui = gr.Dataframe(headers=["Survival", "Final Valuation", "Legacy Headline"], value=[])
            refresh_lb = gr.Button("🔄 Refresh Leaderboard")

        # --- TAB 5: CEO DICTIONARY ---
            gr.Markdown("""
### Reinforcement Learning ↔ Business Terminology

| RL Concept | Business Equivalent | Description |
|---|---|---|
| `reset()` | Company Founding | Start with $10K capital, 20 employees |
| `step()` | Fiscal Quarter | One quarter of operations |
| `state()` | P&L + Sentiment Report | Full company snapshot |
| `action` | CEO Decision | 8 strategic levers |
| Positive Reward | Growth & Stability | Profit, satisfaction, reputation |
| Negative Reward | Risk & Decline | Losses, low morale, crises |
| Policy | CEO Strategy | How the AI picks actions |
| Episode | Company Lifecycle | From founding to exit |

### 🎛️ The 8 Action Levers
1. **Pricing** — Adjust product price up (+)/down (-)
2. **Marketing** — Spend on customer acquisition (+)
3. **Hire/Fire** — Grow (+) or shrink (-) workforce
4. **R&D** — Invest in innovation (+)
5. **Salary** — Raise (+) or cut (-) employee pay
6. **Task Allocation** — Generalize (+) vs specialize (-) teams
7. **Crisis Response** — Invest through (+) vs emergency cut (-)
8. **Budget Shift** — Aggressive growth (+) vs conservative savings (-)

### 🚨 Crisis Types
- **💸 Cash Crisis**: Reserves below $2,000
- **😟 Morale Crisis**: Employee morale below 35%
- **📦 Inventory Crisis**: Stock below 50 units
- **📉 Reputation Crisis**: Brand reputation below 25%
            """)

    # --- Wire Events ---
    def run_auto(task_id, fast):
        env.reset(task_id=task_id)
        yield from stream_simulation(task_id, "auto", speed="fast" if fast else "normal")

    def run_manual(task_id, p, m, h, r, s, t, c, b):
        yield from stream_simulation(task_id, "manual", speed="normal", user_p=p, user_m=m, user_h=h, user_r=r, user_s=s, user_t=t, user_c=c, user_b=b)

    def get_mentor_advice():
        agent = CorporateAgent()
        a_np = agent.compute_action(env.typed_state().to_observation().to_array())
        advice = f"AI MENTOR REASONING:\n1. Pricing: {'Increase' if a_np[0]>0 else 'Decrease'} ({abs(a_np[0]*100):.0f}%)\n"
        advice += f"2. Workforce: {'Expand' if a_np[2]>0 else 'Shrink'} ({abs(a_np[2]*5):.0f} people)\n"
        advice += f"3. Strategy: {'Aggressive Growth' if a_np[7]>0 else 'Conservative Safety'}\n"
        advice += "\nRationale: High confidence in current market trend. Focus on scaling R&D while monitoring cash burn."
        return advice

    def do_export():
        csv_data = export_to_csv(env.typed_state().metrics_history)
        if csv_data:
            with open("ceo_report.csv", "w") as f: f.write(csv_data)
            return "ceo_report.csv"
        return None

    def refresh_leaderboard():
        return leaderboard_data

    outputs_auto = [plot_3d, reward_plot, comp_plot, val_plot, metrics_md, thought_box, actions_box, event_box, roster_table, ticker, mission_report]
    # Order must match stream_simulation yield: ..., headline_html, report_md
    outputs_manual = [m_plot3d, m_reward, m_comp, m_val, m_metrics, m_thought, m_actions, m_events, m_roster, ticker, m_report]

    reset_btn.click(fn=reset_environment, inputs=[task_select], outputs=outputs_auto)
    m_reset_btn.click(fn=reset_environment, inputs=[m_task_select], outputs=outputs_manual)
    
    run_btn.click(fn=run_auto, inputs=[task_select, ff_mode], outputs=outputs_auto)
    manual_btn.click(fn=run_manual, inputs=[m_task_select, mp, mm, mh, mr, ms, mt, mc, mb], outputs=outputs_manual)
    mentor_btn.click(fn=get_mentor_advice, outputs=[mentor_box])
    export_btn.click(fn=do_export, outputs=[csv_output])
    clash_btn.click(fn=run_clash, inputs=[c_p, c_m, c_h, mr, ms, mt, mc, mb], outputs=[clash_plot, clash_msg])
    refresh_lb.click(fn=refresh_leaderboard, outputs=[leaderboard_ui])


# Mount Gradio app to FastAPI server without Authentication
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
