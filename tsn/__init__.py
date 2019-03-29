# flake8: noqa
from catalyst.contrib.registry import Registry
from .experiment import Experiment
from .tsn import tsn

Registry.model(tsn)
