"""Optional helper for calling a local LLM through Ollama (no extra packages).

Any function llm_fn(prompt) -> text works with SpectRAG. This file gives one ready-made:
    llm_fn = ollama_llm("gemma3:4b")
"""

import json
import re
import urllib.request

from spectrag.errors import LLMError
from spectrag.utils.tokens import count_tokens

THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_reply(text):
    """Remove the <think> ... </think> part that reasoning models (e.g. qwen3) print first."""
    return THINKING.sub("", text).strip()


def context_size(prompt, answer_tokens=1024):
    """Ollama's num_ctx for this prompt. Ollama's default window (2-4k tokens) silently drops
    the start of longer prompts, so a 12k-token context needs a bigger window."""
    needed = int(count_tokens(prompt) * 1.15) + answer_tokens  # 15% margin: other tokenizers
    size = 4096
    while size < needed:
        size *= 2
    return size


def post_json(url, body, timeout, retries):
    """POST JSON and return the decoded reply, trying again after a timeout or a dropped
    connection (a busy local model can be slow). Raises LLMError if every try fails."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except OSError as error:  # timeouts and connection errors are both OSError
            if attempt == retries:
                raise LLMError("Ollama request to %s failed after %d tries: %s"
                               % (url, retries + 1, error)) from error


def ollama_llm(model="gemma3:4b", url="http://localhost:11434", num_ctx=None, timeout=600,
               max_tokens=None, retries=2):
    """An llm_fn that calls a local Ollama model with temperature 0 (like the paper).

    num_ctx: the context window in tokens; None sizes it to each prompt.
    max_tokens: the longest answer in tokens; None means no limit.
    retries: how many times to try again after a timeout or a connection error.
    """
    def llm_fn(prompt):
        options = {"temperature": 0, "num_ctx": num_ctx or context_size(prompt)}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        reply = post_json(url.rstrip("/") + "/api/generate", {"model": model, "prompt": prompt,
                          "stream": False, "options": options}, timeout, retries)
        return clean_reply(reply["response"])
    return llm_fn
