# smartroute/providers.py
import os
from typing import List, Dict, Optional, Any

class BaseProvider:
    def generate(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        temp = kwargs.get("temperature", 0.3)
        max_tokens = kwargs.get("max_tokens", 2048)
        resp = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""

class GroqProvider(OpenAIProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url="https://api.groq.com/openai/v1")

class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def generate(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        system = [m["content"] for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        
        resp = self.client.messages.create(
            model=model,
            system="\n\n".join(system) if system else None,
            messages=user_msgs,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.3)
        )
        return "".join([block.text for block in resp.content if hasattr(block, 'text')])
