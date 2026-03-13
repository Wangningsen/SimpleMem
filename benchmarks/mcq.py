"""
Utilities for closed-ended MCQ prompting and answer parsing.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional


_ANSWER_RE = re.compile(
    r"\b(?:final\s+)?answer(?:\s+is)?\s*[:\-]?\s*\(?\s*(?:option\s*)?([A-Z])\s*\)?",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(r"\boption\s*([A-Z])\b", re.IGNORECASE)
_PAREN_RE = re.compile(r"\(([A-Z])\)")
_STANDALONE_RE = re.compile(r"\b([A-Z])\b")


def build_mcq_prompt(question: str, options: Dict[str, str], context: str) -> str:
    """
    Build LongHealthMem-specific MCQ answer prompt.
    """
    ordered_labels = sorted(options.keys())
    options_text = "\n".join(f"{label}. {options[label]}" for label in ordered_labels)
    allowed = ", ".join(ordered_labels)

    return f"""
You are answering a medical-memory multiple-choice question using only the retrieved memory context.

Retrieved Context:
{context}

Question:
{question}

Options:
{options_text}

Instructions:
1. Use only the retrieved context above.
2. Choose exactly one option letter.
3. Output exactly one line in this format: Final answer: <LETTER>
4. <LETTER> must be one of: {allowed}
5. Do not output any additional text.
""".strip()


def parse_mcq_choice(raw_output: str, valid_labels: Iterable[str]) -> Optional[str]:
    """
    Parse model output into a normalized option label.

    Supports outputs like:
    - A
    - Option A
    - (A)
    - Final answer: A
    - Final answer is Option A
    """
    if not raw_output:
        return None

    valid = {label.strip().upper() for label in valid_labels if label and label.strip()}
    if not valid:
        return None

    text = raw_output.strip()
    if not text:
        return None

    if len(text) == 1 and text.upper() in valid:
        return text.upper()

    for regex in (_ANSWER_RE, _OPTION_RE, _PAREN_RE):
        match = regex.search(text)
        if match:
            candidate = match.group(1).upper()
            if candidate in valid:
                return candidate

    for match in _STANDALONE_RE.finditer(text.upper()):
        candidate = match.group(1)
        if candidate in valid:
            return candidate

    return None
