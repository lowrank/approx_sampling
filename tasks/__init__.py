from tasks.base import DownstreamTask, TASK_REGISTRY, get_task_class, register_task
from tasks.downstream import FunctionApproximation, PoissonPINN, DeepRitz

__all__ = [
    "DownstreamTask",
    "TASK_REGISTRY",
    "FunctionApproximation",
    "PoissonPINN",
    "DeepRitz",
    "get_task_class",
    "register_task",
]
