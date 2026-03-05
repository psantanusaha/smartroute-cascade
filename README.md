# SmartRoute: A Training-Free Semantic Router for LLM Cascades (Research)

This repository contains the experiments, datasets, and results for **SmartRoute**, a proactive semantic routing strategy for LLM cascades.

> **Looking for the Python library?** Visit the official repository: [psantanusaha/smartroute](https://github.com/psantanusaha/smartroute)

---

## 📊 Research Summary

SmartRoute demonstrates that by using a "Small" LLM (Llama 3.1 8B) as a **Semantic Bouncer** to pre-classify tasks, developers can achieve frontier-level intelligence at half the cost.

### **Key Findings:**
- **Cost Savings:** **46.94%** reduction in total API spend.
- **Quality:** **4.43 / 5.00** average quality (measured against Sonnet 3.5).
- **Signal Strength:** Zero-shot semantic classification provides **100% recall** on failures, outperforming heuristics (0%) and self-ratings (50%).

---

## 🧪 Experiments

This repository includes the scripts used to generate the data for the SmartRoute whitepaper:

1.  **`experiment_e2_heuristics.py`**: Benchmarking simple refusal/length-based escalation.
2.  **`experiment_e2_self_rating.py`**: Benchmarking LLM self-confidence as an escalation signal.
3.  **`experiment_e2_classifier.py`**: Benchmarking our proactive skill-based routing.
4.  **`experiment_e3_pareto.py`**: Generating the final Cost-Quality Pareto frontier.

### **Reproducing the Results:**
1. Clone this repo.
2. Setup your `.env` with `GROQ_API_KEY` and `ANTHROPIC_API_KEY`.
3. Run the Pareto experiment:
```bash
python experiment_e3_pareto.py --provider groq --verbose
```
4. Analyze the skills:
```bash
python analyze_skills.py
```

---

## 📂 Dataset
- `conversations.json`: 5 sessions of routine tasks (Easy).
- `conversations_hard.json`: 5 sessions of complex boundary tasks (TSP, Quant-Finance, Clinical Reasoning).

---

## 📄 Academic Artifacts
- **[PAPER_OUTLINE.md](PAPER_OUTLINE.md)**: Structured outline for the arXiv preprint.
- **[BIBLIOGRAPHY.bib](BIBLIOGRAPHY.bib)**: BibTeX references for related work (FrugalGPT, RouteLLM, etc.).

---

## 📜 License
MIT
