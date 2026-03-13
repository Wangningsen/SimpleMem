"""
LongHealthMem dataset adapter.

This module isolates benchmark-specific schema parsing and deterministic
turn construction for memory ingestion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from models.memory_entry import Dialogue


@dataclass(frozen=True)
class LongHealthText:
    text_id: str
    text: str
    global_start: Optional[int] = None
    global_end: Optional[int] = None


@dataclass(frozen=True)
class LongHealthQuestion:
    question_id: str
    question: str
    options: Dict[str, str]
    correct_letter: str
    correct_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LongHealthPatientSample:
    patient_id: str
    patient_name: Optional[str]
    birthday: Optional[str]
    diagnosis: Optional[str]
    texts: List[LongHealthText]
    questions: List[LongHealthQuestion]


_TEXT_KEY_RE = re.compile(r"^text_(\d+)$")
_OPTION_FIELDS = ("answer_a", "answer_b", "answer_c", "answer_d", "answer_e")


def _sort_text_keys(keys: Iterable[str]) -> List[str]:
    def _key_order(key: str) -> tuple:
        match = _TEXT_KEY_RE.match(key)
        if match:
            return (0, int(match.group(1)))
        return (1, key)

    return sorted(keys, key=_key_order)


def _parse_text_item(text_id: str, value: Any) -> LongHealthText:
    if isinstance(value, dict):
        text = str(value.get("text", "") or "")
        global_start = value.get("global_start")
        global_end = value.get("global_end")
        return LongHealthText(
            text_id=text_id,
            text=text,
            global_start=global_start if isinstance(global_start, int) else None,
            global_end=global_end if isinstance(global_end, int) else None,
        )

    return LongHealthText(text_id=text_id, text=str(value or ""))


def _extract_options(question_data: Dict[str, Any]) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for option_field in _OPTION_FIELDS:
        option_text = question_data.get(option_field)
        if option_text is None:
            continue
        label = option_field.split("_")[-1].upper()
        options[label] = str(option_text)
    return options


def _resolve_correct_letter(
    question_data: Dict[str, Any],
    options: Dict[str, str],
) -> Optional[str]:
    correct_letter = str(question_data.get("correct_letter", "") or "").strip().upper()
    if correct_letter and correct_letter in options:
        return correct_letter

    correct_text = str(question_data.get("correct", "") or "").strip()
    if not correct_text:
        return None

    for label, option_text in options.items():
        if option_text.strip() == correct_text:
            return label
    return None


def _parse_questions(question_list: List[Dict[str, Any]]) -> List[LongHealthQuestion]:
    parsed_questions: List[LongHealthQuestion] = []
    for idx, question_data in enumerate(question_list):
        question_text = str(question_data.get("question", "") or "").strip()
        if not question_text:
            continue

        options = _extract_options(question_data)
        correct_letter = _resolve_correct_letter(question_data, options)
        if not options or not correct_letter:
            continue

        metadata = {
            "ambiguous_correct": bool(question_data.get("ambiguous_correct", False)),
            "answer_location": question_data.get("answer_location"),
            "evidence_stats": question_data.get("evidence_stats"),
        }
        parsed_questions.append(
            LongHealthQuestion(
                question_id=str(question_data.get("No", idx)),
                question=question_text,
                options=options,
                correct_letter=correct_letter,
                correct_answer=str(question_data.get("correct", "") or "") or None,
                metadata=metadata,
            )
        )

    return parsed_questions


def load_longhealthmem_dataset(
    file_path: Union[str, Path],
) -> List[LongHealthPatientSample]:
    """
    Load LongHealthMem benchmark data and normalize into adapter dataclasses.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"LongHealthMem dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if not isinstance(raw_data, dict):
        raise ValueError("Expected LongHealthMem JSON to be a patient-id keyed object")

    patients: List[LongHealthPatientSample] = []
    for patient_id in sorted(raw_data.keys()):
        patient_data = raw_data[patient_id]
        if not isinstance(patient_data, dict):
            continue

        raw_texts = patient_data.get("texts") or {}
        parsed_texts: List[LongHealthText] = []
        if isinstance(raw_texts, dict):
            for text_key in _sort_text_keys(raw_texts.keys()):
                parsed_texts.append(_parse_text_item(text_key, raw_texts[text_key]))
        elif isinstance(raw_texts, list):
            for idx, value in enumerate(raw_texts):
                parsed_texts.append(_parse_text_item(f"text_{idx}", value))

        questions = _parse_questions(patient_data.get("questions") or [])

        patients.append(
            LongHealthPatientSample(
                patient_id=str(patient_id),
                patient_name=str(patient_data.get("name", "") or "") or None,
                birthday=str(patient_data.get("birthday", "") or "") or None,
                diagnosis=str(patient_data.get("diagnosis", "") or "") or None,
                texts=parsed_texts,
                questions=questions,
            )
        )

    return patients


def chunk_text_by_chars(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Deterministically split text by character length.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized_text = text or ""
    if not normalized_text:
        return []
    if len(normalized_text) <= chunk_size:
        return [normalized_text]

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(normalized_text):
        end = min(start + chunk_size, len(normalized_text))
        chunk = normalized_text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized_text):
            break
        start += step

    return chunks


def convert_patient_to_dialogues(
    patient: LongHealthPatientSample,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    speaker: str = "patient",
) -> List[Dialogue]:
    """
    Convert one patient's longitudinal documents into user-only dialogue turns.

    Default behavior preserves one input text as one dialogue turn.
    Optional character chunking is only applied as a fallback when `chunk_size > 0`
    and a single text exceeds that size.
    """
    dialogues: List[Dialogue] = []
    dialogue_id = 1

    for text_item in patient.texts:
        text = text_item.text or ""
        if not text:
            continue

        chunks = [text]
        if chunk_size and chunk_size > 0 and len(text) > chunk_size:
            chunks = chunk_text_by_chars(
                text=text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

        for chunk in chunks:
            dialogues.append(
                Dialogue(
                    dialogue_id=dialogue_id,
                    speaker=speaker,
                    content=chunk,
                    timestamp=None,
                )
            )
            dialogue_id += 1

    return dialogues


