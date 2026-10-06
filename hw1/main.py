from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from src.data.classes.settings import DEFAULT_COMPONENT, MODEL_ID, ExperimentSettings, RunSpec  # noqa: E402
from src.utils.env import load_env  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
APP_NAME = "hw1"
OUTPUT_DIR_NAME = "outputs"
FINETUNE_OUTPUT_SUFFIX = "pissa_benchmark"
REPORT_OUTPUT_SUFFIX = "report"
SPECTRUM_OUTPUT_SUFFIX = "q4_spectrum"
RUN_SEPARATOR = ":"
DEFAULT_RANK = 16
SUMMARY_COLUMNS = (
    "method",
    "rank",
    "component",
    "gsm8k_accuracy",
    "math_accuracy",
    "gsm8k_correct",
    "gsm8k_total",
    "math_correct",
    "math_total",
    "train_minutes",
    "micro_batch_size",
    "init_seconds",
    "trainable_parameters",
    "gsm8k_unparsed",
    "math_unparsed",
)
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERRUPTED = 130


def parse_run_spec(text: str) -> RunSpec:
    rank_text, _, component = text.partition(RUN_SEPARATOR)
    try:
        rank = int(rank_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid rank in {text!r}; expected RANK[:COMPONENT]") from error
    if rank <= 0:
        raise argparse.ArgumentTypeError(f"rank must be positive: {text!r}")
    return RunSpec(
        rank=rank,
        component=component or DEFAULT_COMPONENT,
    )


def existing_file(text: str) -> Path:
    path = Path(text)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {text}")
    return path


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text!r}")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text!r}")
    return value


def build_output_dir(*, suffix: str) -> Path:
    return ROOT / OUTPUT_DIR_NAME / f"{date.today():%Y-%m-%d}_{suffix}"


def check_gpu_memory(*, min_free_gb: float) -> bool:
    import torch

    from src.utils.cuda_memory import read_free_memory_gb

    if not torch.cuda.is_available():
        logger.error("CUDA is not available; check CUDA_VISIBLE_DEVICES and the torch build.")
        return False
    free_gb = read_free_memory_gb(device_index=0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    logger.info("🧠 Training GPU (CUDA_VISIBLE_DEVICES=%s): %.1f GB free", visible, free_gb)
    if free_gb < min_free_gb:
        logger.error(
            "❌ GPU %s has only %.1f GB free (need %.0f GB); pick a free GPU with nvidia-smi.",
            visible,
            free_gb,
            min_free_gb,
        )
        return False
    return True


def log_environment() -> None:
    import torch

    logger.info("🧠 PyTorch %s, CUDA available=%s", torch.__version__, torch.cuda.is_available())
    for index in range(torch.cuda.device_count()):
        logger.info("🧠 GPU %d: %s", index, torch.cuda.get_device_name(index))


def prepare_huggingface() -> None:
    from src.services.huggingface_auth import build_token_source, login_huggingface

    logger.info("Env file: %s", load_env(start=ROOT))
    log_environment()
    login_huggingface(token_source=build_token_source())


def build_finetune_settings(*, args: argparse.Namespace) -> ExperimentSettings:
    import torch

    return ExperimentSettings(
        result_root=args.output_dir or build_output_dir(suffix=FINETUNE_OUTPUT_SUFFIX),
        train_limit=args.train_limit,
        max_eval_examples=args.max_eval_examples,
        use_data_parallel=torch.cuda.device_count() >= 2,
        save_predictions=not args.no_save_predictions,
        save_adapters=not args.no_save_adapters,
        save_full_model=not args.no_save_full_model,
        oom_retries=args.oom_retries,
        micro_batch_size=args.micro_batch_size,
    )


def run_finetune(args: argparse.Namespace) -> int:
    from src.services.factory import build_experiment_runner, build_range_selectors

    run_specs = args.runs or [RunSpec(rank=DEFAULT_RANK)]
    available_components = sorted(build_range_selectors())
    unknown_components = sorted({spec.component for spec in run_specs} - set(available_components))
    if unknown_components:
        logger.error("Unknown component(s): %s; available: %s", unknown_components, available_components)
        return EXIT_USAGE_ERROR

    prepare_huggingface()
    if not check_gpu_memory(min_free_gb=args.min_free_gpu_gb):
        return EXIT_RUNTIME_ERROR
    settings = build_finetune_settings(args=args)
    logger.debug("Settings: %s", settings)
    logger.info("Effective batch: %d", settings.effective_batch)
    logger.info("MetaMathQA train limit: %s", settings.train_limit)
    logger.info("Evaluation tasks: %s", list(settings.eval_tasks))
    logger.info("📁 Results: %s", settings.result_root)
    logger.info("PiSSA runs: %s", [f"r{spec.rank}:{spec.component}" for spec in run_specs])

    results = build_experiment_runner(settings=settings).run_all(
        run_specs=run_specs,
        run_fresh_baseline=not args.skip_baseline,
    )
    print_summary(results=results)
    return EXIT_OK


def run_report(args: argparse.Namespace) -> int:
    from src.services.report import ReportWriter, build_report_tables, load_successful_results

    output_dir = args.output_dir or build_output_dir(suffix=REPORT_OUTPUT_SUFFIX)
    logger.info("📥 Results: %s", [str(path) for path in args.results])
    results = load_successful_results(paths=args.results)
    tables = build_report_tables(results=results)
    ReportWriter(output_dir=output_dir).write(tables=tables)
    for table in (tables.accuracy, tables.training_time, tables.component_accuracy):
        if table is not None:
            print(table.to_string(), end="\n\n")
    return EXIT_OK


def run_spectrum(args: argparse.Namespace) -> int:
    from src.services.spectrum_analysis import SpectrumSettings, build_spectrum_service

    prepare_huggingface()
    settings = SpectrumSettings(
        output_dir=args.output_dir or build_output_dir(suffix=SPECTRUM_OUTPUT_SUFFIX),
        layer_indices=tuple(args.layers),
        energy_threshold_percent=args.energy_threshold,
        plot_limit=args.plot_limit,
    )
    logger.debug("Settings: %s", settings)
    logger.info("📁 Results: %s", settings.output_dir)
    spectra = build_spectrum_service(
        settings=settings,
        model_id=MODEL_ID,
    ).run()

    print("layer\tminimum_rank\tfull_rank")
    for spectrum in spectra:
        print(f"{spectrum.layer_index}\t{spectrum.minimum_rank}\t{len(spectrum.singular_values)}")
    return EXIT_OK


def print_summary(*, results: pd.DataFrame) -> None:
    frame = results[results["status"].eq("ok")] if "status" in results.columns else results
    columns = [column for column in SUMMARY_COLUMNS if column in frame.columns]
    if len(frame) and columns:
        ordered = frame[columns].sort_values(
            ["method", "rank", "component"],
            na_position="first",
        )
        print(ordered.to_string(index=False))


def add_finetune_parser(*, subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "finetune",
        help="Q1-Q3: PiSSA fine-tune Llama-3.2-1B on MetaMathQA, evaluate on GSM8K + MATH.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=parse_run_spec,
        default=None,
        metavar="RANK[:COMPONENT]",
        help=(
            "PiSSA run, repeatable (e.g. --run 16 --run 16:p25). "
            "COMPONENT: default (top), p25, p50, bottom. "
            f"Default: {DEFAULT_RANK}:{DEFAULT_COMPONENT}."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip evaluating the original model (RUN_FRESH_BASELINE=False).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Result directory. Default: {OUTPUT_DIR_NAME}/<YYYY-MM-DD>_{FINETUNE_OUTPUT_SUFFIX}.",
    )
    parser.add_argument(
        "--train-limit",
        type=positive_int,
        default=25000,
        help="Number of MetaMathQA rows used for training.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=positive_int,
        default=1000,
        help="Number of test rows evaluated per dataset.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=positive_int,
        default=8,
        help="Micro batch per step. Gradient accumulation keeps the global batch at 128.",
    )
    parser.add_argument(
        "--oom-retries",
        type=int,
        default=2,
        help="On CUDA OOM, retry a run up to N times, halving micro_batch_size each time (global batch stays 128).",
    )
    parser.add_argument(
        "--min-free-gpu-gb",
        type=positive_float,
        default=20.0,
        help="Abort before loading data if the training GPU has less free memory than this (GB).",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Do not write per-example prediction JSONL files.",
    )
    parser.add_argument(
        "--no-save-adapters",
        action="store_true",
        help="Do not save trained PiSSA A/B matrices.",
    )
    parser.add_argument(
        "--no-save-full-model",
        action="store_true",
        help="Do not save the merged full model (about 2.5 GB per run).",
    )
    parser.set_defaults(handler=run_finetune)


def add_report_parser(*, subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "report",
        help="Q1-Q3: build accuracy / training-time tables and the Q3 bar chart from benchmark_results.csv.",
    )
    parser.add_argument(
        "--results",
        type=existing_file,
        nargs="+",
        required=True,
        help="One or more benchmark_results.csv files produced by `finetune`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Report directory. Default: {OUTPUT_DIR_NAME}/<YYYY-MM-DD>_{REPORT_OUTPUT_SUFFIX}.",
    )
    parser.set_defaults(handler=run_report)


def add_spectrum_parser(*, subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "spectrum",
        help="Q4: SVD of q_proj weights of the original model, minimum rank for the energy threshold, 6 charts.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 7, 15],
        help="Decoder layer indices to analyse.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=positive_float,
        default=50.0,
        help="Cumulative energy threshold in percent.",
    )
    parser.add_argument(
        "--plot-limit",
        type=positive_int,
        default=200,
        help="Number of leading singular values shown in the charts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Result directory. Default: {OUTPUT_DIR_NAME}/<YYYY-MM-DD>_{SPECTRUM_OUTPUT_SUFFIX}.",
    )
    parser.set_defaults(handler=run_spectrum)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Big Data HW1: SVD analysis for fine-tuning LLMs.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG logs on the console.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    add_finetune_parser(subparsers=subparsers)
    add_report_parser(subparsers=subparsers)
    add_spectrum_parser(subparsers=subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = setup_logging(
        app_name=f"{APP_NAME}_{args.command}",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger.debug("Arguments: %s", vars(args))
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED
    except Exception as error:
        logger.debug("Unhandled error", exc_info=True)
        logger.error("%s (details: %s)", error, log_file)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
