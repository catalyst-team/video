# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from catalyst.dl import SupervisedRunner as Runner
from .tsn import tsn

registry.Model(tsn)
