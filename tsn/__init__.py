# flake8: noqa
from catalyst.contrib.registry import Registry
from .experiment import Experiment
from catalyst.dl.experiments.runner import SupervisedRunner as Runner
from .tsn import tsn

Registry.model(tsn)
