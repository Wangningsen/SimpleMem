"""
LongHealthMem benchmark tester.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.longhealthmem_adapter import (
    LongHealthPatientSample,
    convert_patient_to_dialogues,
    load_longhealthmem_dataset,
)
from benchmarks.mcq import build_mcq_prompt, parse_mcq_choice
from main import SimpleMemSystem


class LongHealthMemTester:
    """
    Evaluate SimpleMem on the LongHealthMem benchmark using MCQ accuracy.
    """

    def __init__(
        self,
        system: SimpleMemSystem,
        dataset_path: str,
        chunk_size: int,
        chunk_overlap: int = 0,
        user_speaker: str = "patient",
    ):
        self.system = system
        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.user_speaker = user_speaker

        self.memory_build_times: List[float] = []
        self.retrieval_times: List[float] = []
        self.answer_times: List[float] = []
        self.total_question_times: List[float] = []

    def load_dataset(self, limit: Optional[int] = None) -> List[LongHealthPatientSample]:
        print(f"Loading LongHealthMem dataset from {self.dataset_path}...")
        samples = load_longhealthmem_dataset(self.dataset_path)
        if limit is not None:
            samples = samples[:limit]
            print(f"Limited to {limit} patients")
        return samples

    def _answer_mcq(self, question: str, options: Dict[str, str], context_str: str) -> str:
        prompt = build_mcq_prompt(question=question, options=options, context=context_str)
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer multiple-choice questions from memory context and follow output "
                    "format strictly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return self.system.llm_client.chat_completion(
            messages=messages,
            temperature=0.0,
            response_format=None,
            max_retries=3,
        )

    def _test_patient(self, sample: LongHealthPatientSample) -> List[dict]:
        dialogues = convert_patient_to_dialogues(
            patient=sample,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            speaker=self.user_speaker,
        )

        print(
            f"\n[Patient {sample.patient_id}] Building memory from "
            f"{len(sample.texts)} documents -> {len(dialogues)} pseudo-turns"
        )

        build_start = time.time()
        self.system.add_dialogues(dialogues)
        self.system.finalize()
        build_time = time.time() - build_start
        self.memory_build_times.append(build_time)

        patient_results: List[dict] = []
        for idx, qa in enumerate(sample.questions):
            print(f"  [Q{idx + 1}] {qa.question}")

            retrieval_start = time.time()
            contexts = self.system.hybrid_retriever.retrieve(qa.question)
            retrieval_time = time.time() - retrieval_start

            context_str = (
                self.system.answer_generator.format_contexts(contexts)
                if contexts
                else "No relevant memory entries were retrieved."
            )

            answer_start = time.time()
            raw_answer = self._answer_mcq(
                question=qa.question,
                options=qa.options,
                context_str=context_str,
            )
            answer_time = time.time() - answer_start
            total_time = retrieval_time + answer_time

            predicted_letter = parse_mcq_choice(raw_answer, qa.options.keys())
            is_correct = predicted_letter == qa.correct_letter

            self.retrieval_times.append(retrieval_time)
            self.answer_times.append(answer_time)
            self.total_question_times.append(total_time)

            patient_results.append(
                {
                    "patient_id": sample.patient_id,
                    "patient_name": sample.patient_name,
                    "question_id": qa.question_id,
                    "question": qa.question,
                    "options": qa.options,
                    "correct_letter": qa.correct_letter,
                    "predicted_letter": predicted_letter,
                    "is_correct": is_correct,
                    "raw_response": raw_answer,
                    "retrieved_context_count": len(contexts),
                    "retrieval_time": retrieval_time,
                    "answer_time": answer_time,
                    "total_time": total_time,
                    "ambiguous_correct": bool(qa.metadata.get("ambiguous_correct", False)),
                }
            )

            print(
                f"    predicted={predicted_letter}, correct={qa.correct_letter}, "
                f"match={is_correct}, retrieved={len(contexts)}"
            )

        return patient_results

    @staticmethod
    def _accuracy(correct: int, total: int) -> float:
        return float(correct / total) if total else 0.0

    def _build_summary(self, results: List[dict], num_patients: int) -> dict:
        total_questions = len(results)
        correct = sum(1 for row in results if row.get("is_correct"))

        per_patient: Dict[str, dict] = {}
        for row in results:
            patient_id = row["patient_id"]
            stats = per_patient.setdefault(patient_id, {"correct": 0, "total": 0})
            stats["total"] += 1
            if row.get("is_correct"):
                stats["correct"] += 1

        for patient_id, stats in per_patient.items():
            stats["accuracy"] = self._accuracy(stats["correct"], stats["total"])

        ambiguous_rows = [row for row in results if row.get("ambiguous_correct")]
        ambiguous_correct = sum(1 for row in ambiguous_rows if row.get("is_correct"))

        return {
            "num_patients": num_patients,
            "num_questions": total_questions,
            "num_correct": correct,
            "accuracy": self._accuracy(correct, total_questions),
            "num_ambiguous_questions": len(ambiguous_rows),
            "ambiguous_accuracy": self._accuracy(ambiguous_correct, len(ambiguous_rows)),
            "avg_memory_build_time": (
                sum(self.memory_build_times) / len(self.memory_build_times)
                if self.memory_build_times
                else 0.0
            ),
            "avg_retrieval_time": (
                sum(self.retrieval_times) / len(self.retrieval_times)
                if self.retrieval_times
                else 0.0
            ),
            "avg_answer_time": (
                sum(self.answer_times) / len(self.answer_times) if self.answer_times else 0.0
            ),
            "avg_total_question_time": (
                sum(self.total_question_times) / len(self.total_question_times)
                if self.total_question_times
                else 0.0
            ),
            "per_patient_accuracy": per_patient,
        }

    def run_test(
        self,
        num_samples: Optional[int] = None,
        save_results: bool = True,
        result_file: str = "longhealthmem_test_results.json",
    ) -> List[dict]:
        print("\n" + "=" * 80)
        print(" SimpleMem LongHealthMem Evaluation ".center(80))
        print("=" * 80 + "\n")

        samples = self.load_dataset(limit=num_samples)
        all_results: List[dict] = []

        for sample in samples:
            self.system.vector_store.clear()
            all_results.extend(self._test_patient(sample))

        summary = self._build_summary(all_results, num_patients=len(samples))

        print("\n" + "=" * 80)
        print(" LongHealthMem Summary ".center(80))
        print("=" * 80)
        print(f"Questions: {summary['num_questions']}")
        print(f"Correct:   {summary['num_correct']}")
        print(f"Accuracy:  {summary['accuracy']:.4f}")

        if save_results:
            output_path = Path(result_file)
            payload = {
                "benchmark": "longhealthmem",
                "chunking": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "speaker": self.user_speaker,
                },
                "summary": summary,
                "detailed_results": all_results,
            }
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")

        return all_results

