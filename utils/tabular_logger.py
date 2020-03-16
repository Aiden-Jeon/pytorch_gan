#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import copy
import numbers

import pandas as pd
import warnings
import torch

from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOutputHandler,
    global_step_from_engine,
)

__all__ = ["TabularLogger", "TabularOutputHandler"]


class TabularOutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.
    Args:
        tag (str): common title for all produced plots. For example, 'training'
        metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, 'another_loss': loss2}` to label the plot
            with corresponding keys.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided,
            uses function output as global_step. To setup global step from another engine, please use
            :meth:`~ignite.contrib.handlers.mlflow_logger.global_step_from_engine`.
    Note:
        Example of `global_step_transform`:
        .. code-block:: python
            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)
    """

    def __init__(self, tag, metric_names=None, output_transform=None,  global_step_transform=None):
        super().__init__(tag, metric_names, output_transform, None, global_step_transform)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TabularLogger):
            raise RuntimeError(f"Handler {self.__class__} works only with TabularLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        rendered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                rendered_metrics["{} {}".format(self.tag, key)] = value
            elif isinstance(value, torch.Tensor) and value.ndimension() == 0:
                rendered_metrics["{} {}".format(self.tag, key)] = value.item()
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    rendered_metrics["{} {} {}".format(self.tag, key, i)] = v.item()
            else:
                warnings.warn("TabularLogger output_handler can not log " "metrics value type {}".format(type(value)))

        logger.log_metrics(metrics, step=global_step, event=event_name, tag=self.tag)


class TabularLogger(BaseLogger):
    """
    Tracking client handler to log parameters and metrics during the training
    and validation, and export the logs as Pandas DataFrame.
    Args:
        None
    """

    def __init__(self):
        self._index = []
        self._records = []
        self.tags = set([])

    def log_metrics(self, metrics, step, event, tag):
        index = {}
        index['timestamp'] = pd.Timestamp.now()
        index['event'] = str(event)
        index['step'] = step
        index['tag'] = tag
        self._index += [index]
        record = copy.deepcopy(metrics)
        self._records += [record]
        self.tags.update([tag])

    def close(self):
        self._records = None

    def dataframe(self, tag=None, index_name='step', metadata=False):
        _records = []
        if tag is None:
            # Prepend tag to the column names, and return the dataframe without setting up index.
            for i, rec in enumerate(self._records):
                _records += [rec.copy()]
                _records[-1].update(dict((k, v) for k, v in self._index[i].items()))
            return pd.DataFrame(_records)
        # tag is not None
        _index = []
        for i, idx in enumerate(self._index):
            if idx['tag'] != tag:
                continue
            _index += [idx[index_name]]
            _records += [self._records[i].copy()]
            if metadata:
                _records[-1].update(dict((k, v) for k, v in idx.items() if k != index_name))
        return pd.DataFrame(_records, index=pd.Index(_index, name=index_name))