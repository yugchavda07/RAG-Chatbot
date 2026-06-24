"""
Ollama Client API Interface
Provides reusable, robust methods to interact with local Ollama service for:
- Checking connection status
- Listing available models
- Generating document embeddings
- Streaming chat completions
"""

import json
from typing import List, Dict, Generator, Any
import requests

class OllamaClient:
    """Handles communication with the local Ollama API service."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def check_connection(self) -> bool:
        """
        Verifies if the Ollama service is running and accessible.
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """
        Fetches the list of all pulled models on the local Ollama instance.
        Returns:
            List[str]: A list of model names.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error listing models: {e}")
        return []

    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        Generates embedding vector for a single string input.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed for model '{model}': {e}")

    def get_embeddings_batch(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of strings.
        Uses the batch `/api/embed` endpoint if available, falling back to individual calls.
        """
        # Try the batch API (/api/embed)
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["embeddings"]
        except Exception:
            # Fallback to serial api/embeddings if /api/embed is unsupported or errors
            pass

        embeddings = []
        for text in texts:
            embeddings.append(self.get_embedding(text, model))
        return embeddings

    def chat_completion_stream(self, 
                               model: str, 
                               messages: List[Dict[str, str]], 
                               options: Dict[str, Any] = None) -> Generator[str, None, None]:
        """
        Streams chat completions from Ollama.
        Args:
            model (str): Name of the chat LLM to run.
            messages (List[Dict[str, str]]): List of chat history messages (role, content).
            options (Dict[str, Any]): Additional LLM parameters (temperature, num_predict, etc.)
        Yields:
            str: Next token chunk from the LLM.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        if options:
            payload["options"] = options

        try:
            # Send streaming POST request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    # Yield content token if present
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                    if chunk.get("done", False):
                        break
        except Exception as e:
            raise RuntimeError(f"Ollama chat completion failed for model '{model}': {e}")
