# based on https://catalyst-team.github.io/catalyst/_modules/catalyst/loggers/wandb.html

from typing import Optional, Dict

try:
    import wandb
except ImportError:
    wandb = None

class Logger:
    """Base class for all loggers."""

    def __init__(self, *args, **kwargs):
        pass

    def log_hparams(self, hparams: Dict) -> None:
        """Logs hyperparameters to the logger."""
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix="train",
    ) -> None:
        """Logs batch and epoch metrics to the logger."""
        pass

    def close_log(self) -> None:
        """Closes the logger."""
        pass

class WandbLogger(Logger):
    """Wandb logger for parameters, metrics, images and other artifacts.

    W&B documentation: https://docs.wandb.com

    Args:
        Project: Name of the project in W&B to log to.
        name: Name of the run in W&B to log to.
        config: Configuration Dictionary for the experiment.
        entity: Name of W&B entity(team) to log to.
        kwargs: Optional,
            additional keyword arguments to be passed directly to the wandb.init
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.project = project
        self.name = name
        self.entity = entity
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            allow_val_change=True,
            **kwargs,
        )

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.run

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, prefix="train"
    ):
        for key, value in metrics.items():
            self.run.log({f"{prefix}/{key}": value}, step=step)

    def log_hparams(self, hparams: Dict) -> None:
        """Logs hyperparameters to the logger."""
        self.run.config.update(hparams)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix="train",
    ) -> None:
        """Logs batch and epoch metrics to wandb."""
        metrics = {k: float(v) for k, v in metrics.items()}
        self._log_metrics(
            metrics=metrics,
            step=step,
            prefix=prefix,
        )

    def close_log(self) -> None:
        """Closes the logger."""
        self.run.finish()