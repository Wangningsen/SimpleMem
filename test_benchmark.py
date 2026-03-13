"""
Unified benchmark entrypoint for SimpleMem.

- benchmark=locomo: uses the existing LoCoMo tester implementation.
- benchmark=longhealthmem: uses LongHealthMem adapter + MCQ evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from benchmarks.longhealthmem_tester import LongHealthMemTester
from main import SimpleMemSystem


def _default_dataset(benchmark: str) -> str:
    if benchmark == "longhealthmem":
        return "benchmark_longmemory_v1.json"
    return "test_ref/locomo10.json"


def _default_result_file(benchmark: str) -> str:
    if benchmark == "longhealthmem":
        return "longhealthmem_test_results.json"
    return "locomo10_test_results.json"


def run_locomo(args: argparse.Namespace, system: SimpleMemSystem):
    # Lazy import to avoid LoCoMo-only heavy dependencies when running LongHealthMem.
    from test_locomo10 import LoCoMoTester

    tester = LoCoMoTester(
        system,
        args.dataset,
        use_llm_judge=args.llm_judge,
        test_workers=args.test_workers,
    )
    return tester.run_test(
        num_samples=args.num_samples,
        save_results=not args.no_save,
        result_file=args.result_file,
        enable_parallel_questions=args.parallel_questions,
    )


def run_longhealthmem(args: argparse.Namespace, system: SimpleMemSystem):
    tester = LongHealthMemTester(
        system=system,
        dataset_path=args.dataset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        user_speaker=args.user_speaker,
    )
    return tester.run_test(
        num_samples=args.num_samples,
        save_results=not args.no_save,
        result_file=args.result_file,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimpleMem benchmark evaluation")

    parser.add_argument(
        "--benchmark",
        type=str,
        default=getattr(config, "BENCHMARK", "locomo"),
        choices=["locomo", "longhealthmem"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to benchmark dataset (default depends on benchmark)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples/patients to evaluate (default: all)",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="Output JSON file (default depends on benchmark)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save result JSON",
    )

    # LoCoMo options
    parser.add_argument(
        "--parallel-questions",
        action="store_true",
        help="(LoCoMo) Process questions in parallel",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="(LoCoMo) Enable LLM-as-judge",
    )
    parser.add_argument(
        "--test-workers",
        type=int,
        default=None,
        help="(LoCoMo) Max worker threads for question processing",
    )

    # LongHealthMem options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=getattr(config, "LONGHEALTHMEM_CHUNK_SIZE", 2000),
        help="(LongHealthMem) Character chunk size for patient text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=getattr(config, "LONGHEALTHMEM_CHUNK_OVERLAP", 200),
        help="(LongHealthMem) Character overlap between chunks",
    )
    parser.add_argument(
        "--user-speaker",
        type=str,
        default=getattr(config, "LONGHEALTHMEM_USER_SPEAKER", "patient"),
        help="(LongHealthMem) Speaker name for synthetic user-only turns",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = _default_dataset(args.benchmark)
    if args.result_file is None:
        args.result_file = _default_result_file(args.benchmark)

    if args.benchmark == "longhealthmem" and args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    dataset_path = Path(args.dataset)
    print("=" * 80)
    print(f"Running benchmark: {args.benchmark}")
    print(f"Dataset: {dataset_path}")
    print("=" * 80)

    system = SimpleMemSystem(clear_db=True)

    if args.benchmark == "locomo":
        run_locomo(args, system)
    else:
        run_longhealthmem(args, system)


if __name__ == "__main__":
    main()

