# llm.py
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def run_llm1(prompt, model="gpt-oss", stream=False):
    """
    Sends a prompt to the Ollama API and returns the generated response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    # If streaming is off, expect a single JSON with 'response'
    return response

def run_llm2(prompt, model="gemma3:12b", stream=False):
    """
    Sends a prompt to the Ollama API and returns the generated response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    res_json = response.json()
    return res_json["response"]

def run_llm3(prompt, model="codegemma", stream=False):
    """
    Sends a prompt to the Ollama API and returns the generated response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response

if __name__ == "__main__":
    # For testing only
    prompt = "Say hello from Ollama!"
    print(run_llm1(prompt))
