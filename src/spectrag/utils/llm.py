"""Optional helper for calling a local LLM through Ollama (no extra packages).

Any function llm_fn(prompt) -> text works with SpectRAG. This file gives one ready-made:
    llm_fn = ollama_llm("gemma3:4b")
"""

import json
import re
import urllib.request

THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_reply(text):
    """Remove the <think> ... </think> part that reasoning models (e.g. qwen3) print first."""
    return THINKING.sub("", text).strip()


def ollama_llm(model="gemma3:4b", url="http://localhost:11434"):
    """An llm_fn that calls a local Ollama model with temperature 0 (like the paper)."""
    def llm_fn(prompt):
        body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0}}).encode("utf-8")
        request = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return clean_reply(json.loads(response.read())["response"])
    return llm_fn
