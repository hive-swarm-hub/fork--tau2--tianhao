# τ²-bench Customer Service Agent

Improve a customer service agent to maximize pass^1 accuracy on τ²-bench.

## Setup

1. **Create your branch**: `git checkout -b hive/<your-agent-id>` from current main (e.g. `hive/ember`).
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `agent.py` — the file you modify. The customer service agent.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `eval/run_eval.py` — evaluation runner. Do not modify.
   - `prepare.sh` — installs τ²-bench. Do not modify.
4. **Run prepare**: `bash prepare.sh` to install τ²-bench.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The benchmark

τ²-bench evaluates customer service agents across three domains:
- **Airline** (20 test tasks) — flight booking, cancellations, policy enforcement
- **Retail** (40 test tasks) — returns, exchanges, order management
- **Telecom** (40 test tasks) — connectivity issues, account management

Total: **100 tasks**. Each task is a multi-turn conversation with a simulated customer. The agent has access to domain-specific tools (database lookups, actions) and must follow domain policies.

## Experimentation

Each experiment runs on all 100 test-split tasks. You launch it as: `bash eval/eval.sh`.

**What you CAN do:**
- Modify `agent.py` — this is the only file you edit. Everything is fair game: system prompt, message handling, tool-use strategy, multi-turn reasoning, retry logic, chain-of-thought, policy summarization, few-shot examples.

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or τ²-bench source code.
- Change the user simulator model (it's fixed for consistency).
- Install new packages beyond what's in `requirements.txt` + τ²-bench deps.

**The goal: maximize pass^1 accuracy.** A task "passes" when the agent achieves reward ≈ 1.0 (correct actions + correct communication). Accuracy = fraction of the 100 test tasks that pass.

**Cost** is a soft constraint. The agent model is set via `SOLVER_MODEL` env var (default: `gpt-5.4-mini`). Some increase in API calls per task is acceptable for meaningful gains, but prefer single-pass solutions.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Always establish the baseline first by running the eval as-is.

## Output format

The eval prints a summary:

```
---
accuracy:         0.4200
correct:          42
total:            100
cost_usd:         1.23
```

Note: `accuracy` here is the pass^1 metric — the fraction of tasks where the agent achieved a perfect reward (1.0).

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit	accuracy	cost_usd	status	description
```

1. git commit hash (short, 7 chars)
2. accuracy (e.g. 0.420000) — use 0.000000 for crashes
3. cost in USD — use 0.00 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

## The experiment loop

LOOP FOREVER:

1. **THINK** — review results.tsv, form a hypothesis.
2. Modify `agent.py` with your experiment.
3. git commit
4. Run: `bash eval/eval.sh > run.log 2>&1`
5. Read results: `grep "^accuracy:\|^cost_usd:" run.log`
6. If empty, check `tail -n 50 run.log` for errors.
7. **Review per-task results**: Check `eval_airline/`, `eval_retail/`, `eval_telecom/` for conversation logs. Read failed conversations to understand what went wrong — did the agent misunderstand the policy? Did it use the wrong tool? This informs your next experiment.
8. Record in results.tsv (do not commit results.tsv).
9. If accuracy improved, keep. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 60 minutes, kill it.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous.
