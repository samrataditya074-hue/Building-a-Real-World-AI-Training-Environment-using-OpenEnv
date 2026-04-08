"""
demo_nontech.py — A plain-English demonstration of the CEO AI Simulator.

This script runs one episode (20 quarters) and prints a human-readable
story of what happened each quarter — no charts, no technical jargon.

Designed for: hackathon judges, non-technical stakeholders, and anyone
who wants to understand what the AI is actually doing.

Usage:
    python demo_nontech.py
    python demo_nontech.py --quarters 10 --seed 99
"""

from __future__ import annotations

import argparse
import os
import sys
import io

# Force UTF-8 output for Windows console compatibility
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CEOEnvironment
from agent.business_agent import CorporateAgent
from models import Action


# ── Pretty formatting helpers ─────────────────────────────────────────────────

# --- NEW: Utility function for safe RL output scaling ---
def scale_positive(x: float) -> float:
    """Safely map RL outputs from [-1.0, 1.0] to [0.0, 1.0] to satisfy Pydantic >=0 constraints."""
    return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))

def _currency(val: float) -> str:
    """Formats a number as $1,234 or -$1,234."""
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.0f}"


def _pct(val: float, total: float = 100.0) -> str:
    """Formats a value as a percentage string."""
    return f"{val / total * 100:.0f}%"


def _bar(val: float, max_val: float = 100.0, width: int = 20) -> str:
    """Renders a simple ASCII progress bar."""
    filled = int(round((val / max_val) * width))
    filled = max(0, min(filled, width))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _describe_action(action: Action, state_cash: float, state_morale: float) -> str:
    """
    Converts the 8 raw action values into a plain-English sentence
    describing what the AI decided to do this quarter.
    """
    decisions = []

    # Pricing
    if action.price_adjustment > 0.2:
        decisions.append(f"raised the product price by ${action.price_adjustment * 8:.1f}")
    elif action.price_adjustment < -0.2:
        decisions.append(f"lowered the product price by ${abs(action.price_adjustment) * 8:.1f} to attract more customers")

    # Marketing
    spend = action.marketing_push * 600
    if spend > 200:
        decisions.append(f"spent {_currency(spend)} on marketing campaigns")
    elif spend < 50:
        decisions.append("held back on marketing to save costs")

    # Hiring / Firing
    people = int(round(action.hire_fire * 5))
    if people > 0:
        decisions.append(f"hired {people} new employee{'s' if people > 1 else ''}")
    elif people < 0:
        decisions.append(f"let go of {abs(people)} low-performing employee{'s' if abs(people) > 1 else ''}")

    # R&D
    rd = action.rd_investment * 400
    if rd > 100:
        decisions.append(f"invested {_currency(rd)} in research & development")

    # Salary
    if action.salary_adjustment > 0.03:
        decisions.append(f"gave top performers a {action.salary_adjustment * 10:.0f}% salary bonus")
    elif action.salary_adjustment < -0.03:
        decisions.append(f"cut salaries by {abs(action.salary_adjustment) * 10:.0f}% to reduce costs")

    # Team structure
    if action.task_allocation > 0.3:
        decisions.append("cross-trained teams to make them more flexible")
    elif action.task_allocation < -0.3:
        decisions.append("had teams specialise in their core strengths")

    # Crisis response (only mention if it was triggered)
    if action.crisis_response > 0.2 and state_cash < 50_000:
        decisions.append("invested extra cash to address company crises")
    elif action.crisis_response < -0.3 and state_morale < 60:
        decisions.append("made emergency cuts to preserve cash")

    # Budget
    if action.budget_shift > 0.3:
        decisions.append("moved the budget towards aggressive growth")
    elif action.budget_shift < -0.3:
        decisions.append("shifted to a conservative, savings-focused budget")

    if not decisions:
        return "maintained the current strategy without major changes."

    # Join them naturally
    if len(decisions) == 1:
        return decisions[0] + "."
    elif len(decisions) == 2:
        return f"{decisions[0]} and {decisions[1]}."
    else:
        return f"{', '.join(decisions[:-1])}, and {decisions[-1]}."


def _describe_trend(profit: float, prev_profit: float) -> str:
    """Describes whether things improved or worsened this quarter."""
    delta = profit - prev_profit
    if delta > 2_000:
        return "📈 Strong improvement this quarter!"
    elif delta > 0:
        return "↗ Things improved slightly."
    elif delta > -2_000:
        return "↘ Small step back this quarter."
    else:
        return "📉 Significant drop — things got harder."


def _crisis_warning(s_dict: dict) -> str:
    """Returns a warning string if any crisis is active, otherwise empty."""
    crises = []
    if s_dict.get("cash_crisis"):
        crises.append("💸 Cash reserves are critically low!")
    if s_dict.get("morale_crisis"):
        crises.append("😟 Employee morale is dangerously low — people may quit!")
    if s_dict.get("inventory_crisis"):
        crises.append("📦 Stock is running out — we're losing sales!")
    if s_dict.get("reputation_crisis"):
        crises.append("📉 Brand reputation is damaged — customers are leaving!")
    return "  ⚠️  " + " | ".join(crises) if crises else ""


def _educational_tip(quarter: int, s_dict: dict) -> str:
    """Returns a rotating educational tip about the simulation mechanics."""
    tips = [
        "💡 R&D is like planting seeds: invest now, harvest efficiency gains for many quarters.",
        "💡 Firing people saves on payroll, but damages the morale of everyone who stays.",
        "💡 A price lower than the competitor's attracts more customers but earns less per sale.",
        "💡 Marketing spend directly translates into more customer demand this quarter.",
        "💡 Brand reputation grows slowly over time — it's hard to build but easy to damage.",
        "💡 The market trend (economy) is outside the AI's control — it must adapt to it.",
        "💡 Employee morale affects their performance: happy teams work better and earn more.",
        "💡 Cash below $2,000 triggers a crisis — the AI shifts to emergency survival mode.",
        "💡 Cross-training teams makes them flexible; specialisation makes them very efficient.",
        "💡 R&D above 70% starts boosting brand reputation automatically every quarter.",
    ]
    return tips[(quarter - 1) % len(tips)]


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(quarters: int, seed: int) -> None:
    """
    Run one episode and print a plain-English story of each quarter.
    """
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        🏢  AUTONOMOUS CEO AI SIMULATOR — LIVE DEMO          ║")
    print("║           Watching the AI run a company for you             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Seed: {seed} | Quarters: {quarters}")
    print("  The AI will make all decisions. Watch what happens.\n")
    print("─" * 66)

    # Set up environment and agent
    env = CEOEnvironment()
    agent = CorporateAgent()
    obs = env.reset(seed=seed)

    prev_profit = 0.0
    final_valuation = 0.0

    for q in range(1, quarters + 1):
        # AI decides
        obs_array = obs.to_array()
        action_np = agent.compute_action(obs_array)
        action = Action(
            price_adjustment=float(action_np[0]),
            marketing_push=scale_positive(action_np[1]), # mapped [-1, 1] -> [0, 1]
            hire_fire=float(action_np[2]),
            rd_investment=scale_positive(action_np[3]),  # mapped [-1, 1] -> [0, 1]
            salary_adjustment=float(action_np[4]),
            task_allocation=float(action_np[5]),
            crisis_response=float(action_np[6]),
            budget_shift=float(action_np[7]),
        )

        obs = env.step(action)
        s = env.state()          # dict — OpenEnv API
        ts = env.typed_state()   # typed State — for rich data

        profit_sign = "+" if ts.profit >= 0 else ""
        trend_emoji = "🟢" if ts.profit > 0 else ("🔴" if ts.profit < -500 else "🟡")

        print(f"\n{'─' * 66}")
        print(f"  QUARTER {q:02d}  {trend_emoji}  {ts.headline}")
        print(f"{'─' * 66}")

        # Core financials
        print(f"\n  📊 FINANCIALS")
        print(f"     Cash     : {_currency(ts.cash):>12}   {_bar(min(ts.cash, 400_000), 400_000)}")
        print(f"     Revenue  : {_currency(ts.revenue):>12}   (this quarter)")
        print(f"     Profit   : {profit_sign}{_currency(ts.profit):>12}")

        # People & brand
        print(f"\n  👥 PEOPLE & BRAND")
        print(f"     Employees: {ts.total_employees:>4} people")
        print(f"     Morale   : {ts.employee_morale:>5.1f}%  {_bar(ts.employee_morale)}")
        print(f"     Customers: {ts.customer_satisfaction:>5.1f}%  {_bar(ts.customer_satisfaction)}")
        print(f"     Brand    : {ts.brand_reputation:>5.1f}%  {_bar(ts.brand_reputation)}")

        # Technology
        print(f"\n  🔬 TECHNOLOGY")
        print(f"     R&D      : {ts.rd_progress:>5.1f}%  {_bar(ts.rd_progress)}")
        print(f"     Efficiency: {ts.operational_efficiency:>5.1f}% {_bar(ts.operational_efficiency)}")

        # What the AI decided
        action_sentence = _describe_action(action, ts.cash, ts.employee_morale)
        print(f"\n  🤖 AI DECISION")
        print(f"     The AI {action_sentence}")

        # Trend from last quarter
        print(f"\n  {_describe_trend(ts.profit, prev_profit)}")

        # Crisis warnings (if any)
        crisis_msg = _crisis_warning(s)
        if crisis_msg:
            print(f"\n{crisis_msg}")

        # Educational tip (rotates each quarter)
        print(f"\n  {_educational_tip(q, s)}")

        final_valuation = ts.get_valuation()
        prev_profit = ts.profit

        if obs.done:
            if ts.cash <= 0:
                print(f"\n  ☠️  SIMULATION TERMINATED: BANKRUPTCY in Quarter {q}!")
            elif ts.customer_satisfaction <= 5:
                print(f"\n  ☠️  SIMULATION TERMINATED: CUSTOMER EXODUS in Quarter {q}!")
            elif ts.total_employees <= 2:
                print(f"\n  ☠️  SIMULATION TERMINATED: WORKFORCE COLLAPSED in Quarter {q}!")
            break

    # Final summary
    print(f"\n{'═' * 66}")
    print(f"  🏁  SIMULATION COMPLETE — {q} QUARTERS")
    print(f"{'═' * 66}")
    print(f"  Final Cash       : {_currency(env.typed_state().cash)}")
    print(f"  Final Valuation  : {_currency(final_valuation)}")
    print(f"  Final Employees  : {env.typed_state().total_employees}")
    print(f"  Final Morale     : {env.typed_state().employee_morale:.1f}%")
    print(f"  Final R&D Level  : {env.typed_state().rd_progress:.1f}%")
    print()
    print("  To see the full interactive dashboard with live charts, run:")
    print("  python -m uvicorn server.app:app --host 0.0.0.0 --port 7860")
    print(f"{'═' * 66}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plain-English CEO AI demo — no charts, no jargon, just storytelling."
    )
    parser.add_argument(
        "--quarters", type=int, default=20,
        help="Number of quarters to simulate (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible results (default: 42)"
    )
    args = parser.parse_args()
    run_demo(quarters=args.quarters, seed=args.seed)


if __name__ == "__main__":
    main()
