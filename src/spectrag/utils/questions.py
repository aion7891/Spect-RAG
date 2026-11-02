"""Utility questions: questions a chunk can answer (paper Section 3.2.1, "Utility").

The paper asks gpt-4o-mini for them. Here you bring any LLM as a function
llm_fn(prompt) -> text, and make_question_fn() turns it into a question generator.
"""

import json
import re
import urllib.request

QUESTION_PROMPT = """Read the passage and write {count} short questions that the passage answers.
Write one question per line. Do not number them. Do not add anything else.

Passage:
{text}"""


def parse_questions(reply):
    """Keep the lines that look like questions, without numbering or bullets."""
    questions = []
    for line in reply.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line.endswith("?"):
            questions.append(line)
    return questions


def make_question_fn(llm_fn, count=3):
    """Return question_fn(text) -> list of up to 'count' questions."""
    def question_fn(text):
        reply = llm_fn(QUESTION_PROMPT.format(count=count, text=text))
        return parse_questions(reply)[:count]
    return question_fn


def add_utility_questions(chunks, question_fn):
    """Store the generated questions in every chunk as chunk['utility_questions']."""
    for chunk in chunks:
        chunk["utility_questions"] = question_fn(chunk["text"]) if question_fn else []
    return chunks


def ollama_llm(model="gemma3:4b", url="http://localhost:11434"):
    """Optional helper: an llm_fn that calls a local Ollama model (temperature 0, like the paper)."""
    def llm_fn(prompt):
        body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0}}).encode("utf-8")
        request = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["response"]
    return llm_fn
