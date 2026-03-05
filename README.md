# SmartRoute: A Training-Free Semantic Router for LLM Cascades

SmartRoute is an open-source library that helps developers reduce LLM costs by **47%** without sacrificing intelligence. It uses a "Small" LLM (like Llama 3.1 8B) as a **Semantic Bouncer** to pre-classify tasks and route them to the most cost-effective tier.

---

## 🚀 Key Features

- **Cost Savings:** Up to 47% reduction in API spend compared to always-frontier models.
- **High Quality:** Maintains 4.43/5.00 average quality (vs 5/5 baseline).
- **Proactive Routing:** Zero-shot skill classification catches failures *before* they happen.
- **Multi-Provider:** Support for Groq, Anthropic, and OpenAI.
- **Training-Free:** Works out-of-the-box with zero preference logs or datasets needed.

---

## 📦 Installation

```bash
pip install git+https://github.com/psantanusaha/smartroute-cascade.git
```

---

## 💻 Quick Start

```python
import os
from smartroute import SmartRouter

# Define your tiers (mix-and-match providers)
router = SmartRouter(
    cheap_config={
        "provider": "groq", 
        "model": "llama-3.1-8b-instant", 
        "api_key": os.getenv("GROQ_API_KEY")
    },
    expensive_config={
        "provider": "anthropic", 
        "model": "claude-3-5-sonnet-20240620", 
        "api_key": os.getenv("ANTHROPIC_API_KEY")
    }
)

# One API to route them all
result = router.generate("Design a thread-safe LRU cache in Python.")

print(f"Routed to: {result['model']} (Detected Skill: {result['skill']})")
print(f"Response: {result['response']}")
```

---

## 📊 The "SmartRoute" Signal (Research Results)

We compared three ways to decide when to escalate a prompt:

| Signal Type | Recall (Caught Fails) | Overheads |
| :--- | :--- | :--- |
| **Heuristics** | 0% | $0 |
| **Self-Rating** | 50% | Low |
| **SmartRoute Classifier** | **100%** | **Low** |

**The Result:** By pre-classifying intent, SmartRoute provides **"Semantic Safety"**—identifying complex tasks (like Quant-Finance or Multi-step Logic) and escalating them immediately, while keeping routine tasks on the 100x cheaper tier.

---

## 🛠️ Advanced: Custom Taxonomy

You can override the default routing logic by providing your own skill-to-tier mapping:

```python
custom_taxonomy = {
    "factual_qa": "expensive", # Always use the big model for facts
    "creative_simple": "cheap" 
}
router = SmartRouter(..., taxonomy=custom_taxonomy)
```

---

## 🔗 Related Research
- **FrugalGPT (Stanford):** Focuses on confidence thresholds.
- **RouteLLM (LMSys):** Uses trained classifiers on preference data.
- **Our Contribution:** First training-free, proactive semantic router optimized for Mixed-API cascades.

---

## 📜 License
MIT
