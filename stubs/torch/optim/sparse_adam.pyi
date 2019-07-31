# Stubs for torch.optim.sparse_adam (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from .optimizer import Optimizer
from typing import Any, Optional

class SparseAdam(Optimizer):
    def __init__(self, params: Any, lr: float = ..., betas: Any = ..., eps: float = ...) -> None: ...
    def step(self, closure: Optional[Any] = ...): ...