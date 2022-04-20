from functools import partial

import ConfigSpace as CS
from braindecode.models import TCN

from brain_decode_project.benchmarks.compound_benchmark import ComposedConfigurationSpace, \
    ComposedBenchmark
from brain_decode_project.data import TUHData
from brain_decode_project.data.tuh_data import TUHDataSplitter
from brain_decode_project.modules.budget_module import DummyBudgetManager
from brain_decode_project.pl_modules import RegressionModule

"""
This example shows how to create a class dynamically. This is useful when starting the
experiments from a config file.
"""

def factory():

    attrs = dict(
        data_set_type=partial(
            TUHData,
            data_target_name='age',
            cut_off_first_k_seconds=60,
            n_max_minutes=3,
            sfreq=100,
            rng=0,
            train_or_eval='both',  # train, eval, both
            only_healthy=False,
            standardization_op='exponential_moving_demean',
        ),

        data_set_splitter_type=partial(
            TUHDataSplitter,
            input_window_samples=1600
        ),

        network_type=partial(
            TCN,
            n_outputs=1,
            add_log_softmax=False,
        ),

        lightning_model_type=RegressionModule,

        config_space=ComposedConfigurationSpace(
            configuration_spaces=[
                CS.ConfigurationSpace()
            ]
        ),

        budget_manager_type=DummyBudgetManager,
    )

    CustomClass = type('CustomClass', (ComposedBenchmark, ), attrs)
    return CustomClass