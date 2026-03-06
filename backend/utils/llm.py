import os
import requests

class LLMService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("LLM_MODEL", "tinyllama")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        """
        Returns (answer: str, is_error: bool).
        Callers check is_error to decide whether to fall back to raw context.
        """
        # Cap prompt length to avoid overloading small models like tinyllama
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        if len(combined_prompt) > 3000:
            combined_prompt = combined_prompt[:3000] + "\n\n[Context truncated for model capacity]"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": combined_prompt,
                    "stream": False,
                    "temperature": temperature,
                    "options": {
                        "num_predict": 256,   # limit output tokens for speed
                        "num_ctx": 2048       # context window
                    }
                },
                timeout=180  # tinyllama loads into RAM on first call — allow 3 minutes
            )
            response.raise_for_status()
            return response.json()["response"], False

        except requests.exceptions.ConnectionError:
            return (
                "⚠️ Ollama is not running. Start with `ollama serve` and pull: "
                f"`ollama pull {self.model}`.",
                True
            )
        except requests.exceptions.ReadTimeout:
            return (
                "⚠️ Ollama timed out loading the model. Wait 30 seconds and try again.",
                True
            )
        except Exception as e:
            return f"LLM Error: {str(e)}", True

llm_service = LLMService()
