# SmartRoute: A Training-Free Semantic Router for LLM Cascades
## Paper Outline (arXiv Preprint)

### 1. Abstract
- The problem: Frontier models (GPT-4o, Sonnet 3.5) are expensive; small models (Haiku, Llama 8B) are cheap but fail on complex reasoning.
- Our solution: SmartRoute, a proactive semantic router using zero-shot classification.
- Key results: 47% cost savings, 4.43/5 average quality, 100% recall on failure turns.

### 2. Introduction
- Motivation: The "Intelligence-Cost" trade-off in production LLM apps.
- Limitations of current work: Trained routers (RouteLLM) are rigid; reactive cascades (FrugalGPT) are slow/expensive.
- Our Contribution: A training-free, proactive, model-agnostic routing taxonomy.

### 3. Related Work
- **Reactive Cascades:** FrugalGPT (Confidence thresholds).
- **Trained Routers:** RouteLLM (LMSYS preference data), Hybrid LLM (ICLR '24).
- **Reasoning Cascades:** LLM-Cascade (Mixture of Thoughts, 2310.03094).
- **Benchmarking:** RouterBench (Oracle-based evaluation).

### 4. Methodology: Semantic Skill-Based Routing
- **The 12-Skill Taxonomy:** Categorizing intent from `factual_qa` to `agentic_workflow`.
- **Zero-Shot LLM Classifier:** Using an 8B model as a "Semantic Bouncer".
- **The Cascade Policy:** Proactive escalation for "Hard" skills; Reactive fallback (optional).

### 5. Experimental Setup
- **Models:** Llama-3.1-8B (Cheap) vs Llama-3.3-70B / Claude-3.5-Sonnet (Expensive).
- **Providers:** Groq (High throughput) and Anthropic.
- **Datasets:** 
  - `conversations.json`: 5 sessions of routine/easy tasks.
  - `conversations_hard.json`: 5 sessions of "boundary" tasks (TSP, Quant-Finance, Logic).

### 6. Results and Analysis
- **Experiment E2 (Signal Comparison):** Heuristics vs Self-Rating vs Classifier.
- **Experiment E3 (Cost-Quality Frontier):** Savings vs Quality Pareto analysis.
- **Performance by Skill:** Which skills trigger escalation most? (Heatmap/Table).

### 7. Discussion
- **The "Semantic Safety" vs. "Latency" trade-off.**
- **Model-Agnosticism:** Ease of swapping future models (GPT-5, etc).
- **Mixed-Provider Strategy:** Bypassing rate limits via routing.

### 8. Conclusion
- Summary of impact and future directions (Local SLM routers).

### 9. References
- Citations for all 10+ papers from HANDOFF.md.
