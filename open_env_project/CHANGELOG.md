# CHANGELOG

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — Hackathon Production Release

### Summary
Full OpenEnv compliance pass, continuous reward shaping, three deterministic graders,
non-technical UX, complete test suite, and GitHub Actions CI.

---

### Modified Files

#### `openenv.yaml` (EXPANDED — was 7 lines, now ~120 lines)
**Reason**: The previous version was a stub with only basic metadata.
The OpenEnv spec requires a full schema including `action_space`,
`observation_space`, `tasks`, and `reward` sections.
All 8 action levers and 14 observation fields are now documented with
ranges, normalization constants, and plain-English descriptions.
The 3 tasks are listed with grader references and score formulas.

#### `models.py` (UPGRADED — Pydantic + Reward + inline comments)
**Reason**: `Action` and `Observation` were plain `@dataclass` objects with no
validation. In production, an agent sending out-of-range values (e.g., `marketing_push=-0.5`)
would silently cause wrong business calculations. Now they are `pydantic.BaseModel`
with `ge`/`le` validators that raise `ValidationError` immediately.
Added `Reward` dataclass for structured reward breakdown.
Added plain-English inline comments on every field for non-technical readers.
`State` and `Employee` remain `@dataclass` (they hold mutable lists — Pydantic
does not handle these well without extra config).

#### `tasks.py` (IMPROVED — type hints + docstrings)
**Reason**: Existing graders lacked type annotations and plain-English documentation.
Logic unchanged to preserve backward compatibility with existing tests.
Added full docstrings explaining each task in simple language.

#### `server/environment.py` (MAJOR REWRITE)
**Reason**: Multiple compliance and quality gaps existed:
1. No seed support → episodes were not reproducible.
2. Only sparse end-of-step reward; no per-step improvement signals → slow RL learning.
3. `state()` returned a typed `State` object → OpenEnv spec requires `dict`.
4. No structured logging → hard to debug agent behaviour.
5. No episode trace export → no way to post-analyse what happened.

**What changed**:
- `reset(seed=None)` now seeds `random.Random` and `np.random.default_rng`.
- Added 4 continuous reward shaping components:
  `profit_delta` (quarter-over-quarter profit change),
  `morale_delta` (quarter-over-quarter morale change),
  `rd_payoff` (R&D progress bonus),
  `fire_penalty` (destructive mass-layoff penalty).
- `state()` now returns `dict`; internal typed access moved to `typed_state()`.
- `server/app.py` references updated from `env.state()` → `env.typed_state()`.
- `logging.INFO` emitted for every action decoded, every event triggered,
  every financial summary, every crisis, and every reward.
- `episode_trace` list accumulates per-step dicts; written to JSON when
  `TRACE_LOGGING=1` env var is set (off by default to avoid disk spam).
- All normalization constants exported as module-level `REWARD_NORM_*` names.
- Comprehensive plain-English comments added on every simulation stage.

#### `Dockerfile` (IMPROVED)
**Reason**: Had no way to pass OpenAI API key, no health check, no install of `curl`.
Added `ARG OPENAI_API_KEY` + `ENV` passthrough, `HEALTHCHECK` directive,
`curl` system package for health check, and comments on alternative `CMD` for baseline.

#### `requirements.txt` (EXPANDED)
**Reason**: `openai` was missing (needed for `baseline_inference.py`);
`pandas` was used (`ceo_report.csv` export) but not listed.
Added both with minimum version pins.

#### `README.md` (FULL REWRITE)
**Reason**: The old README was technically dense and inaccessible to non-technical readers.
Rewritten with:
- One-paragraph plain-English summary at the top
- Two cheat sheet tables (8 levers + 14 observation fields)
- Step-by-step commands (install → demo → dashboard → docker → baseline → train → test)
- Tasks & scoring table with plain descriptions and score formulas
- FAQ for non-technical users with a single-command demo
- Reproducibility block with exact commands and expected JSON output
- Full project structure map

#### `tests/test_ceo_env.py` (FIXED + EXPANDED)
**Reason**: Had `assert state.cash == 10000.0` which was wrong — `models.py`
initializes `State.cash = 200_000.0`. Fixed to `200_000`.
Added: seed reproducibility test, `state()` dict format test, full episode smoke test.

---

### New Files

#### `graders.py` (NEW)
**Reason**: The existing `tasks.py` graders operate on the final `State` snapshot.
The OpenEnv task spec requires episode-history-aware graders that can measure
things like "how many quarters was profit positive?" or "was valuation growing
throughout the run?". These require a different function signature
(`episode_history: list[dict]` instead of `state: State`).
Keeping them in a new file avoids breaking `tasks.py` imports.
Contains `grade_easy`, `grade_medium`, `grade_hard`, and `GRADERS` registry.

#### `baseline_inference.py` (NEW)
**Reason**: No existing script combined deterministic seeding + all 3 tasks +
OpenAI API support + JSON output. Required by the hackathon spec.
Falls back to `CorporateAgent` heuristic when no `OPENAI_API_KEY` is set,
so it always works out-of-the-box with zero configuration.

#### `demo_nontech.py` (NEW)
**Reason**: The interactive dashboard requires a browser and many dependencies.
Judges and non-technical stakeholders need a quick, accessible way to see the
AI making decisions with zero setup beyond `pip install`.
Prints quarter-by-quarter plain-English CEO decisions with ASCII bars,
rotating educational tips, crisis warnings, and a final summary.

#### `tests/test_openenv_spec.py` (NEW)
**Reason**: No existing test validated that the Pydantic models reject bad input,
or that `reset()`/`step()`/`state()` follow the OpenEnv API contract.
Now tests: Action validation, Observation array shape, all three API methods.

#### `tests/test_graders.py` (NEW)
**Reason**: Required by spec — graders must be deterministic and tested.
Tests all three graders with healthy/empty/winning/losing episode fixtures,
verifying score ranges, determinism, edge cases, and boundary behaviors.

#### `tests/test_reward_shaping.py` (NEW)
**Reason**: Reward quality is critical for RL training stability. Needed tests
that verify no NaN/Inf rewards, reasonable bounds, terminal penalty activation,
fire_penalty sign correctness, rd_payoff monotonicity, and all components finite.

#### `.github/workflows/ci.yml` (NEW)
**Reason**: No CI existed. Every PR should automatically run tests and the baseline
script. Zero external secrets required (baseline works without API key).

---

### Unchanged Files (intentionally preserved)
- `env/business_env.py` — legacy Gymnasium environment, kept as reference
- `env/business_state.py` — legacy state for Gymnasium env
- `env/control_env.py` — industrial temperature control demo
- `env/state.py` — state for temperature control demo
- `agent/business_agent.py` — unchanged; guardrails and heuristic strategies intact
- `agent/baseline.py` — legacy simple rule-based agent
- `train_rl.py` — RL training script (unchanged; uses CEOEnvironment internally)
- `baseline.py` — legacy benchmark runner (kept for backward compat)
- `main.py` — industrial control demo entry point
- `deploy.ps1` — Hugging Face deployment script
- `ceo_ppo_model.zip` — pre-trained PPO brain (binary, not modified)
