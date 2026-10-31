import json
import requests
from typing import List, Dict, Any, Generator
from config import OLLAMA_BASE_URL, MODEL_NAME

class OllamaClient:
    """Client for interacting with local Ollama instance running phi3.5."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = MODEL_NAME):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama is accessible and whether the model is installed."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                has_model = any(self.model in m for m in models)
                return {
                    "connected": True,
                    "model_found": has_model,
                    "available_models": models,
                    "error": None if has_model else f"Model '{self.model}' not found in Ollama."
                }
            return {
                "connected": False,
                "model_found": False,
                "available_models": [],
                "error": f"Ollama returned HTTP {resp.status_code}"
            }
        except requests.exceptions.RequestException as e:
            return {
                "connected": False,
                "model_found": False,
                "available_models": [],
                "error": f"Could not connect to Ollama at {self.base_url}: {e}"
            }

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, stream: bool = False, max_tokens: int = None) -> Any:
        """
        Send a chat completion request to Ollama.
        If stream=False, gathers the complete streaming chunks to avoid connection read timeouts.
        If stream=True, returns a generator yielding chunks of text.
        """
        options = {
            "temperature": temperature,
            "num_ctx": 2048,       # phi3.5 handles context well; 2048 fits schema + question cleanly
            "num_thread": 8,       # Use all logical CPU threads
            "num_batch": 512,      # Better CPU throughput
            "repeat_penalty": 1.0  # 1.0 is essential for phi3.5; >1.0 corrupts variables & numbers
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # Always stream from Ollama to keep connection active and avoid HTTP ReadTimeout
            "keep_alive": "30m",
            "options": options
        }

        try:
            if stream:
                return self._stream_response(payload)
            else:
                chunks = []
                with requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=300
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line:
                            chunk = json.loads(line.decode("utf-8"))
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                chunks.append(content)
                            if chunk.get("done", False):
                                break
                return "".join(chunks)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def _stream_response(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=300
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break

def test_connection() -> Dict[str, Any]:
    client = OllamaClient()
    return client.check_health()
