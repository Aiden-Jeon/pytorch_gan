from typing import Callable, Union, Any, Optional

import torch.nn as nn
from ignite.engine import Engine


class AbstractGAN(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, batch: Optional[Union[tuple, list]]) -> dict:
        return {"loss": None}

    def fit_batch(self, engine: Engine, batch: Optional[Union[tuple, list]]) -> dict:
        return self.fit(batch)
