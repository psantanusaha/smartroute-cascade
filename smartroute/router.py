# smartroute/router.py
from typing import List, Dict, Optional, Any
from .providers import OpenAIProvider, GroqProvider, AnthropicProvider
from .taxonomy import DEFAULT_SKILL_TAXONOMY, CLASSIFIER_PROMPT

class SmartRouter:
    def __init__(
        self,
        cheap_config: Dict[str, str],
        mid_config: Optional[Dict[str, str]] = None,
        expensive_config: Optional[Dict[str, str]] = None,
        taxonomy: Optional[Dict[str, str]] = None,
    ):
        self.cheap_provider = self._init_provider(cheap_config)
        self.cheap_model = cheap_config["model"]
        
        self.mid_provider = self._init_provider(mid_config) if mid_config else self.cheap_provider
        self.mid_model = mid_config["model"] if mid_config else self.cheap_model
        
        self.expensive_provider = self._init_provider(expensive_config) if expensive_config else self.mid_provider
        self.expensive_model = expensive_config["model"] if expensive_config else self.mid_model
        
        self.taxonomy = taxonomy or DEFAULT_SKILL_TAXONOMY

    def _init_provider(self, config: Dict[str, str]):
        p = config["provider"].lower()
        key = config["api_key"]
        if p == "openai": return OpenAIProvider(api_key=key)
        if p == "groq": return GroqProvider(api_key=key)
        if p == "anthropic": return AnthropicProvider(api_key=key)
        raise ValueError(f"Unknown provider: {p}")

    def classify(self, prompt: str) -> str:
        messages = [{"role": "user", "content": CLASSIFIER_PROMPT.format(prompt=prompt)}]
        resp = self.cheap_provider.generate(self.cheap_model, messages, temperature=0.0, max_tokens=20)
        
        cleaned = resp.strip().lower().replace("-", "_").replace(" ", "_")
        for skill in self.taxonomy.keys():
            if skill in cleaned:
                return skill
        return "ambiguous_open"

    def generate(self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs) -> Dict[str, Any]:
        # 1. Classify
        skill = self.classify(prompt)
        tier = self.taxonomy.get(skill, "mid")
        
        # 2. Select Route
        if tier == "cheap":
            provider, model = self.cheap_provider, self.cheap_model
        elif tier == "mid":
            provider, model = self.mid_provider, self.mid_model
        else:
            provider, model = self.expensive_provider, self.expensive_model
            
        # 3. Generate
        messages = history or []
        messages.append({"role": "user", "content": prompt})
        
        response = provider.generate(model, messages, **kwargs)
        
        return {
            "response": response,
            "skill": skill,
            "tier": tier,
            "model": model,
            "provider": provider.__class__.__name__
        }
