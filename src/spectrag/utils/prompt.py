"""Turn the selected chunks into context text and a grounded prompt (Algorithm 1 line 13).

The paper formats the context "grouped by source document" and uses a synthesis-oriented,
grounded prompt (Table 2, "Generation").
"""

from spectrag.utils.sections import is_heading

ANSWER_PROMPT = """Answer the question using only the sources below.
Combine facts from different sources when the question needs it.
If the sources do not contain the answer, say that you do not know.
Mention the source names you used.

{context}

Question: {question}
Answer:"""


def group_by_document(chunks):
    """Documents ordered by their best chunk score; chunks inside a document in reading order."""
    groups = {}
    for chunk in chunks:
        groups.setdefault(chunk["doc_id"], []).append(chunk)
    ordered = sorted(groups.values(), key=lambda group: -max(c["score"] for c in group))
    return [sorted(group, key=lambda c: c["position"]) for group in ordered]


def format_chunk(chunk):
    """One chunk's text. The section path is added only when the text does not already
    start with its heading (e.g. the 2nd chunk of a long section)."""
    section = " > ".join(chunk["section"])
    first_line = chunk["text"].split("\n", 1)[0]
    if not section or is_heading(first_line):
        return chunk["text"]
    return "[" + section + "]\n" + chunk["text"]


def format_context(chunks):
    """All selected chunks as one text, one block per source document."""
    blocks = []
    for group in group_by_document(chunks):
        body = "\n\n".join(format_chunk(chunk) for chunk in group)
        blocks.append("### Source: " + group[0]["doc_id"] + "\n" + body)
    return "\n\n".join(blocks)


def build_prompt(question, context):
    """The grounded prompt for the generator LLM."""
    return ANSWER_PROMPT.format(context=context, question=question)
