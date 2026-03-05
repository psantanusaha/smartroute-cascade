# SmartRoute / Model Escalation — Handoff Document

## What We're Building

An open-source library + research paper on **LLM cascade routing**:
> Start every request on the cheapest model. Escalate to a more expensive model only when quality drops. Save money without sacrificing quality.

Target: **arXiv preprint / whitepaper** for professional profile.

---

## What's Been Built

### `escalation_experiment.py` — Core experiment script
- Loads conversations from JSON
- For each user turn: calls cheap model + expensive model + judge (expensive model scores cheap response 1-5)
- Marks escalation point (first turn with score < threshold, default 3)
- Calculates cost for: all-expensive vs cascade strategy
- Supports `--provider anthropic | openai | claude-code`
- Flags: `--verbose`, `--output results.json`, `--max-output-tokens`, `--fail-threshold`
- Auto-loads `.env` file (no need to export API keys)

### `conversations.json` — 5 easy test sessions
Simple to medium complexity. All handled by Haiku fully.

### `conversations_hard.json` — 5 hard test sessions
Designed to trigger escalation on the final turn.

### SmartRoute experiments (from Claude Desktop)
- `config.py` — Groq client, 3-tier model config, 12-skill taxonomy, 60 test prompts
- `experiment_1_capability_gaps.py` — Tests all prompts on cheap+mid, maps skill gaps. Now supports `--skills` flag to run one skill at a time
- `experiment_2_traffic_distribution.py` — Realistic traffic simulation on cheap model only
- `experiment_3_skill_classifier.py` — 3 classification approaches: keywords, LLM, TF-IDF
- `analyze_results.py` — Combined verdict across all 3 experiments
- `quick_test.py` — Setup smoke test (skip this, use single API call instead)

---

## API Keys / Setup

`.env` file (already configured):
```
ANTHROPIC_API_KEY=...   # for escalation_experiment.py
GROQ_API_KEY=...        # for SmartRoute experiments (free tier)
```

Run with venv:
```bash
venv/bin/python escalation_experiment.py --provider anthropic --input conversations.json --verbose
venv/bin/python experiment_1_capability_gaps.py --skills factual_qa basic_code
```

---

## Experiment Results So Far

### escalation_experiment.py (Anthropic — Haiku vs Sonnet)

**Easy sessions (conversations.json) — 5 sessions:**
- Cheap model handled all 5 fully (100%)
- Average savings: **~75%** vs always-Sonnet
- All turns scored 4-5/5

**Hard sessions (conversations_hard.json) — 5 sessions:**

| Session | Hard turn score | Escalated? | Savings |
|---|---|---|---|
| algorithms-tsp | 4/5 | No | 73% |
| clinical-reasoning | 4/5 | No | 72% |
| **quant-finance** | **2/5** | **Yes (turn 3)** | **28%** |
| logic-puzzle | 3/5 | No (barely) | 73% |
| collab-doc-design | 4/5 | No | 69% |

**Key finding:** Haiku is hard to break. Only precise multi-constraint numerical code (Monte Carlo pricer with control variates) triggered escalation. Broad knowledge, reasoning, system design — Haiku handles it all at 4+/5.

### SmartRoute Experiments (Groq — Llama 8B vs 70B)

**Experiment 3 — Skill Classifier:**
| Approach | Accuracy | Cost | Latency |
|---|---|---|---|
| Keyword heuristics | 80% | Free | <1ms |
| Cheap LLM (Llama 8B) | 87% | ~$0.001/req | ~200ms |
| TF-IDF cross-val | 52% | Free | <5ms |

→ LLM classifier at 87% is the best approach. Keywords at 80% are a good free fallback.

**Experiment 1 — Capability Gaps:**
- Cheap (Llama 8B) passes 87% of prompts
- Mid (Llama 70B) also passes 87% — no meaningful gap between these two tiers
- `ambiguous_open` — both models score 0% (judge too strict on subjective questions)
- `agentic` — cheap 60%, mid 40% (both struggle)

**Experiment 2 — Traffic Distribution:**
- Results unreliable due to Groq rate limit timeouts mid-run
- Do not use these numbers — re-run with longer pauses or smaller batches

---

## Key Insight for the Paper

**Haiku (3B-class) handles ~90% of requests adequately when judged by Sonnet.**
**Cost savings: 69-76% on sessions that never escalate.**
**The one failure mode: precise, multi-constraint code generation with specific numerical methods.**

This validates the cascade hypothesis clearly.

---

## Related Papers (Download These)

| Paper | arXiv | Priority |
|---|---|---|
| FrugalGPT (Stanford 2023) | arxiv.org/abs/2305.05176 | ⭐ Must read |
| RouteLLM (LMSys 2024) | arxiv.org/abs/2406.18665 | ⭐ Must read |
| Survey: Doing More with Less (2025) | arxiv.org/abs/2502.00409 | ⭐ Must read |
| LLM Cascades + Mixture of Thoughts | arxiv.org/abs/2310.03094 | High |
| Hybrid LLM (ICLR 2024) | arxiv.org/abs/2404.14618 | High |
| RouterBench | arxiv.org/abs/2403.12031 | High |
| Unified Routing + Cascading (ETH 2024) | arxiv.org/abs/2410.10347 | Medium |
| TREACLE (NeurIPS 2024) | arxiv.org/abs/2404.13082 | Medium |
| Faster Cascades + Speculative Decoding | arxiv.org/abs/2405.19261 | Medium |
| EcoAssistant | arxiv.org/abs/2310.03046 | Low |
| Tryage | arxiv.org/abs/2308.11601 | Low |
| IBM LLM Routing | openreview.net/forum?id=Zb0ajZ7vAt | Low |

---

## Paper Positioning

**Existing work:**
- FrugalGPT: cascade with learned confidence thresholds, closed benchmarks (HEADLINES, OVERRULING, COQA)
- RouteLLM: trained classifier on Chatbot Arena preference data, open-source models
- Hybrid LLM: DeBERTa router, tunable threshold, MixInstruct dataset

**Your differentiation:**
- **Training-free** — no classifier to train, no preference data needed
- **Commercial API focus** — Anthropic + OpenAI (not just open-source models)
- **Systematic signal comparison** — first paper to compare heuristic vs self-rating vs classifier vs LLM-judge as escalation signals on commercial APIs
- **Practical open-source library** — drop-in wrapper people can actually use

---

## What's Left to Do

### Immediate (experiments)
1. **Expand E1** — run 200+ prompts across providers (Anthropic + OpenAI), get statistically meaningful baseline for "% traffic handled by cheap model"
2. **E2 — Signal comparison** — implement and compare 4 escalation signals:
   - Heuristic (response length, hedging phrases like "I'm not sure", "I cannot")
   - Cheap model self-rating ("Rate your confidence in this answer 1-5")
   - Keyword pre-classifier on prompt
   - LLM judge (current method — expensive baseline)
3. **E3 — Cost-quality frontier** — sweep fail_threshold from 1→5, plot cost savings % vs quality at each point (the paper's main figure)

### Paper
4. Write related work section (after reading the 3 priority papers)
5. Define your unique contribution clearly against FrugalGPT + RouteLLM
6. Design figures: cost-quality Pareto curve, escalation signal comparison table, per-skill escalation rate heatmap

### Library
7. Design the public API (drop-in wrapper for Anthropic/OpenAI client)
8. Implement pluggable escalation signals
9. Publish on GitHub with README + paper link

---

## Recommended Next Session Start

```
"Let's design experiment E2 — the escalation signal comparison.
We need to implement 4 signals and compare their accuracy vs cost
on the same 200 prompts."
```
