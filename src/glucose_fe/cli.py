from __future__ import annotations
import argparse, logging, os
from .config import PipelineConfig
from .pipeline import run_parallel, compile_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def main() -> None:
    p = argparse.ArgumentParser(description="Glucose Feature Engineering Pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--max-workers", type=int, default=None)
    args = p.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)

    # Pin Polars threads (must happen before any polars import)
    if (cfg.engine or "pandas").lower() == "polars" and cfg.polars_threads:
        os.environ["POLARS_MAX_THREADS"] = str(cfg.polars_threads)
        logging.info("POLARS_MAX_THREADS=%s", os.environ["POLARS_MAX_THREADS"])

    if not args.compile_only:
        run_parallel(cfg, max_workers=args.max_workers)
    out_path = compile_all(cfg)
    logging.info("Compiled dataset -> %s", out_path)

if __name__ == "__main__":
    main()

# Run: python -m src.glucose_fe.cli --config configs/fe_config.yaml --max-workers 1