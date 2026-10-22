from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from src.data.classes.settings import ExperimentSettings  # noqa: E402
from src.services.config_loader import ConfigError, load_experiment_settings, load_graph_settings  # noqa: E402
from src.utils.env import load_env  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.storage import ProjectStorage, StorageNotFoundError, prepare_storage  # noqa: E402

if TYPE_CHECKING:
    from src.services.training import RunPaths

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
APP_NAME = "hw2"
DEFAULT_EXPERIMENT_CONFIG = CONFIG_ROOT / "xgboost" / "exp1.yaml"
DEFAULT_GRAPH_CONFIG = CONFIG_ROOT / "graphs" / "render.yaml"
MIB_PER_GB = 1024
EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERRUPTED = 130


def existing_file(text: str) -> Path:
    path = Path(text)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {text}")
    return path


def optional_existing_file(text: str) -> Path | None:
    if not text.strip():
        return None
    return existing_file(text)


def config_tag(*, config: Path) -> str:
    resolved = config.resolve()
    relative = resolved.relative_to(CONFIG_ROOT) if resolved.is_relative_to(CONFIG_ROOT) else Path(resolved.name)
    return ".".join(relative.with_suffix("").parts)


def run_name(*, config: Path) -> str:
    return f"{date.today():%Y-%m-%d}_{APP_NAME}.{config_tag(config=config)}"


def hw2_folder(
    *,
    storage: ProjectStorage,
    name: str,
) -> Path:
    return storage.folder(name=name) / APP_NAME


def latest_run(
    *,
    model_root: Path,
    config: Path,
) -> str:
    suffix = f"_{APP_NAME}.{config_tag(config=config)}"
    runs = sorted(path.name for path in model_root.glob(f"*{suffix}") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No trained run matching *{suffix} in {model_root}. Run `bash scripts/train.sh` first.")
    return runs[-1]


def check_gpu(*, device: str, min_free_gb: float) -> bool:
    from src.utils.gpu import read_free_memory_mib, visible_gpu_ids

    if not device.startswith("cuda"):
        logger.error("model.device is %r. Keep training on CUDA.", device)
        return False
    gpu_ids = visible_gpu_ids()
    if not gpu_ids:
        logger.error("CUDA_VISIBLE_DEVICES is empty. Set GPU_IDS in the script.")
        return False
    free_gb = read_free_memory_mib(server_gpu_id=gpu_ids[0]) / MIB_PER_GB
    logger.info("🧠 GPU %s (CUDA_VISIBLE_DEVICES=%s): %.1f GB free", gpu_ids[0], ",".join(gpu_ids), free_gb)
    if free_gb < min_free_gb:
        logger.error("GPU %s has %.1f GB free, need %.1f GB. Pick another GPU with nvidia-smi.", gpu_ids[0], free_gb, min_free_gb)
        return False
    return True


def run_download(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.data_download import DatasetDownloader

    settings = load_experiment_settings(path=args.config)
    snapshot_dir = hw2_folder(storage=storage, name="data") / "snapshots" if args.keep_snapshot else None
    DatasetDownloader(settings=settings.data).download(snapshot_dir=snapshot_dir)
    return EXIT_OK


def run_paths(
    *,
    storage: ProjectStorage,
    name: str,
) -> RunPaths:
    from src.services.training import RunPaths

    return RunPaths(
        output_dir=hw2_folder(storage=storage, name="outputs") / name,
        model_dir=hw2_folder(storage=storage, name="models") / name,
    )


def model_device(*, settings: ExperimentSettings) -> str:
    devices = {"rdl_gnn": settings.gnn.device, "tabpfn": settings.tabpfn.device}
    return devices.get(settings.model_type, settings.model.device)


def run_train(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.factory import build_predictor, build_trainer, load_data

    settings = load_experiment_settings(path=args.config)
    if not check_gpu(device=model_device(settings=settings), min_free_gb=args.min_free_gpu_gb):
        return EXIT_RUNTIME_ERROR
    name = args.run_name or run_name(config=args.config)
    paths = run_paths(storage=storage, name=name)
    logger.info("🚀 Experiment %s (%s), run %s", settings.name, settings.model_type, name)
    data = load_data(settings=settings.data)
    build_trainer(
        settings=settings,
        data=data,
        paths=paths,
        cache_dir=None if args.no_cache else hw2_folder(storage=storage, name="cache") / "features",
    ).run(data_end=data.data_end)
    if not args.no_predict:
        build_predictor(
            settings=settings,
            data=data,
            paths=paths,
        ).predict(data_end=data.data_end)
    return EXIT_OK


def run_build_features(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.factory import build_dataset_builder, load_data
    from src.services.sampling import SamplePlanner

    settings = load_experiment_settings(path=args.config)
    data = load_data(settings=settings.data)
    cutoffs = SamplePlanner(settings=settings.sampling).plan_training_cutoffs(data_end=data.data_end)
    build_dataset_builder(
        settings=settings,
        data=data,
        cache_dir=hw2_folder(storage=storage, name="cache") / "features",
    ).build_many(cutoffs=cutoffs)
    logger.info("💾 Weekly feature tables cached for %d cutoffs", len(cutoffs))
    return EXIT_OK


def run_predict(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.factory import build_predictor, load_data

    settings = load_experiment_settings(path=args.config)
    name = args.run_name or latest_run(model_root=hw2_folder(storage=storage, name="models"), config=args.config)
    logger.info("🔮 Predicting with run %s", name)
    data = load_data(settings=settings.data)
    build_predictor(
        settings=settings,
        data=data,
        paths=run_paths(storage=storage, name=name),
    ).predict(data_end=data.data_end)
    return EXIT_OK


def run_evaluate(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.evaluation import SubmissionEvaluator

    settings = load_experiment_settings(path=args.config)
    evaluation_dir = settings.data.evaluation_dir
    truth = None if args.format_only else args.truth
    SubmissionEvaluator(evaluation_dir=evaluation_dir).evaluate(
        prediction=args.pred,
        truth=truth,
    )
    return EXIT_OK


def run_ensemble(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.ensemble import EnsembleBuilder
    from src.services.factory import build_tuner

    settings = load_experiment_settings(path=args.configs[0])
    model_root = hw2_folder(storage=storage, name="models")
    output_root = hw2_folder(storage=storage, name="outputs")
    run_dirs = [output_root / latest_run(model_root=model_root, config=config) for config in args.configs]
    EnsembleBuilder(
        tuner=build_tuner(settings=settings),
        step=args.step,
        evaluation_dir=settings.data.evaluation_dir,
    ).build(
        run_dirs=run_dirs,
        output_dir=output_root / f"{date.today():%Y-%m-%d}_{APP_NAME}.{args.name}",
    )
    return EXIT_OK


def read_submission_run(*, output_root: Path) -> str | None:
    path = output_root / "submission" / "best_run.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def run_benchmark(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.benchmark import OutputBenchmark
    from src.services.training import HOLDOUT_FILE

    settings = load_experiment_settings(path=args.config)
    output_root = hw2_folder(storage=storage, name="outputs")
    run_dirs = sorted(
        path.parent
        for path in output_root.glob(f"*/{HOLDOUT_FILE}")
        if not path.parent.name.startswith(args.skip_prefix)
    )
    OutputBenchmark(
        evaluator=settings.data.evaluation_dir / "evaluate.py",
        model_root=hw2_folder(storage=storage, name="models"),
        submission_run=read_submission_run(output_root=output_root),
    ).run(
        run_dirs=run_dirs,
        output_dir=output_root / f"{date.today():%Y-%m-%d}_{APP_NAME}.benchmark",
    )
    return EXIT_OK


def run_compare(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.services.comparison import RunComparator

    model_root = hw2_folder(storage=storage, name="models")
    output_root = hw2_folder(storage=storage, name="outputs")
    run_dirs = [output_root / latest_run(model_root=model_root, config=config) for config in args.configs]
    run_dirs += sorted(output_root.glob(f"{date.today():%Y-%m-%d}_{APP_NAME}.ensemble*"))
    RunComparator(output_dir=output_root / f"{date.today():%Y-%m-%d}_{APP_NAME}.comparison").compare(
        run_dirs=run_dirs,
        submission_dir=output_root / "submission",
    )
    return EXIT_OK


def run_graphs(args: argparse.Namespace, storage: ProjectStorage) -> int:
    from src.data.loaders.relational_db import ParquetDatabaseLoader
    from src.services.graph_renderer import GraphReport

    settings = load_graph_settings(path=args.config)
    database = ParquetDatabaseLoader(db_dir=settings.data.db_dir).load()
    GraphReport(
        settings=settings,
        database=database,
        output_dir=args.output_dir or hw2_folder(storage=storage, name="outputs") / (args.run_name or run_name(config=args.config)),
    ).run()
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BigData HW2: predict GitHub user activity")
    parser.add_argument("--verbose", action="store_true", help="show debug logs in the console")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="delete the old TA clone and clone the latest release")
    download.add_argument("--config", type=existing_file, default=DEFAULT_EXPERIMENT_CONFIG)
    download.add_argument("--keep-snapshot", action="store_true", help="copy the old data/db to storage before deleting it")
    download.set_defaults(handler=run_download)

    features = subparsers.add_parser("build-features", help="build and cache the weekly samples shared by all models")
    features.add_argument("--config", type=existing_file, default=DEFAULT_EXPERIMENT_CONFIG)
    features.set_defaults(handler=run_build_features)

    train = subparsers.add_parser("train", help="build weekly samples, train 6 classifiers and tune thresholds")
    train.add_argument("--config", type=existing_file, default=DEFAULT_EXPERIMENT_CONFIG)
    train.add_argument("--run-name", default=None)
    train.add_argument("--no-cache", action="store_true", help="rebuild features instead of reading the parquet cache")
    train.add_argument("--min-free-gpu-gb", type=float, default=2.0)
    train.add_argument("--no-predict", action="store_true", help="skip writing output_314540080.csv after training")
    train.set_defaults(handler=run_train)

    predict = subparsers.add_parser("predict", help="write output_314540080.csv for the test users")
    predict.add_argument("--config", type=existing_file, default=DEFAULT_EXPERIMENT_CONFIG)
    predict.add_argument("--run-name", default=None, help="trained run to use (default: the latest run of this config)")
    predict.set_defaults(handler=run_predict)

    evaluate = subparsers.add_parser("evaluate", help="check a prediction CSV with the TA evaluate.py and log the score")
    evaluate.add_argument("--config", type=existing_file, default=DEFAULT_EXPERIMENT_CONFIG)
    evaluate.add_argument("--pred", type=existing_file, required=True)
    evaluate.add_argument("--truth", type=optional_existing_file, default=None, help="answer CSV to score against (empty: only check the format)")
    evaluate.add_argument("--format-only", action="store_true", help="only check the format and the test UserIDs")
    evaluate.set_defaults(handler=run_evaluate)

    ensemble = subparsers.add_parser("ensemble", help="blend the latest runs of several configs, weights and thresholds chosen on calibration")
    ensemble.add_argument("--configs", type=existing_file, nargs="+", required=True)
    ensemble.add_argument("--name", default="ensemble")
    ensemble.add_argument("--step", type=float, default=0.1)
    ensemble.set_defaults(handler=run_ensemble)

    benchmark = subparsers.add_parser("benchmark", help="score every run on the holdout week with the TA evaluate.py and check every output CSV")
    benchmark.add_argument("--config", type=existing_file, default=DEFAULT_EXPERIMENT_CONFIG)
    benchmark.add_argument("--skip-prefix", default="smoke", help="skip run folders starting with this")
    benchmark.set_defaults(handler=run_benchmark)

    compare = subparsers.add_parser("compare", help="compare the latest run of each config and copy the best output")
    compare.add_argument("--configs", type=existing_file, nargs="+", required=True)
    compare.set_defaults(handler=run_compare)

    graphs = subparsers.add_parser("graphs", help="render the Q1 schema graph and relational entity graph")
    graphs.add_argument("--config", type=existing_file, default=DEFAULT_GRAPH_CONFIG)
    graphs.add_argument("--run-name", default=None)
    graphs.add_argument("--output-dir", type=Path, default=None, help="folder for the graph files (default: storage outputs)")
    graphs.set_defaults(handler=run_graphs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = load_env(start=ROOT)
    try:
        storage = prepare_storage(repo_root=ROOT.parent)
    except StorageNotFoundError as error:
        print(f"❌ {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    log_file = setup_logging(
        app_name=f"{APP_NAME}_{args.command}" + (f"_{config_tag(config=args.config)}" if hasattr(args, "config") else ""),
        log_dir=hw2_folder(storage=storage, name="logs"),
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger.info("Env file: %s. Storage: %s. Log: %s", env_file, storage.root, log_file)
    try:
        return args.handler(args, storage)
    except ConfigError as error:
        logger.error("%s", error)
        return EXIT_USAGE_ERROR
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED
    except Exception:
        logger.exception("%s failed", args.command)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
