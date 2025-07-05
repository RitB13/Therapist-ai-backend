import os
import requests
from langchain_core.language_models.llms import LLM
from typing import Optional, List
from dotenv import load_dotenv

# Load .env
load_dotenv()

class GroqLLM(LLM):
    model: str = "llama3-70b-8192"
    temperature: float = 0.7
    api_key: str = os.getenv("GROQ_API_KEY")
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "groq-chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if stop:
            payload["stop"] = stop

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError(f"Groq API Error: {response.status_code} - {response.text}")

# LangChain-compatible factory
def load_llm():
    return GroqLLM()
