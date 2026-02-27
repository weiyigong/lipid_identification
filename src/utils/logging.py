"""Unified training logger with wandb and file backends."""

import json
import logging
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """Logs metrics and messages to either wandb or a local log file."""

    def __init__(self, backend="wandb", project=None, name=None, config=None,
                 log_dir=None, resume=False):
        self.backend = backend
        self.name = name or "train"

        if backend == "wandb":
            import wandb
            self._wandb = wandb
            wandb.init(
                project=project or "lipid-identification",
                name=name,
                config=config or {},
                resume="allow" if resume else None,
            )
        elif backend == "file":
            log_dir = Path(log_dir or "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.name.replace(" ", "_")

            self._metrics_path = log_dir / f"{safe_name}_{ts}_metrics.jsonl"
            self._log_path = log_dir / f"{safe_name}_{ts}.log"

            self._logger = logging.getLogger(f"train.{safe_name}")
            self._logger.setLevel(logging.INFO)
            self._logger.handlers.clear()

            fh = logging.FileHandler(self._log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
            self._logger.addHandler(fh)

            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(sh)

            self._logger.propagate = False
            self._logger.info(f"Config: {json.dumps(config or {}, default=str)}")
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'wandb' or 'file'.")

    def log(self, metrics, step=None):
        """Log a dict of scalar metrics to the metrics JSONL file."""
        if self.backend == "wandb":
            self._wandb.log(metrics, step=step)
        else:
            row = dict(metrics)
            if step is not None:
                row["_step"] = step
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def log_batch(self, metrics, step=None):
        """Log batch-level losses to the log file (not metrics JSONL)."""
        if self.backend == "wandb":
            self._wandb.log(metrics, step=step)
        else:
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items()]
            msg = f"[step {step}] " + " | ".join(parts) if step is not None else " | ".join(parts)
            self._logger.info(msg)

    def log_table(self, key, dataframe):
        """Log a pandas DataFrame."""
        if self.backend == "wandb":
            self._wandb.log({key: self._wandb.Table(dataframe=dataframe)})
        else:
            table_path = self._metrics_path.with_name(
                self._metrics_path.stem + f"_{key}.csv"
            )
            dataframe.to_csv(table_path, index=False)
            self._file_msg(f"Saved table '{key}' to {table_path}")

    def print(self, msg):
        """Print a message and log it."""
        if self.backend == "wandb":
            print(msg)
        else:
            self._logger.info(msg)

    def finish(self):
        """Finalize the logging session."""
        if self.backend == "wandb":
            self._wandb.finish()
        else:
            self._file_msg("Training finished.")
            for h in self._logger.handlers[:]:
                h.close()
                self._logger.removeHandler(h)

    def _file_msg(self, msg):
        self._logger.info(msg)
