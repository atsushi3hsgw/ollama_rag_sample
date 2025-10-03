import requests
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api")

def query_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Send a prompt to the Ollama API and return the full response.
    """
    payload = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post(f"{OLLAMA_URL}/generate", json=payload, stream=True)

    full_text = ""
    for line in response.iter_lines():
        if line:
            chunk = line.decode("utf-8")
            if '"response":"' in chunk:
                # Extract the response text from the JSON chunk
                text = chunk.split('"response":"')[1].split('"')[0]
                full_text += text
    return full_text

import requests

def get_available_models():
    """
    Get a list of available models from the Ollama
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/tags")
        tags = response.json().get("models", [])
        return [tag["name"] for tag in tags]
    except Exception as e:
        return ["llama3"]  # Fallback
