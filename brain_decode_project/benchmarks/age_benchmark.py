from functools import partial

import ConfigSpace as CS

from brain_decode_project.benchmarks.compound_benchmark import ComposedBenchmark, \
    ComposedConfigurationSpace
from brain_decode_project.benchmarks.factory import factory
from brain_decode_project.data import TUHData
from brain_decode_project.data.tuh_data import TUHDataSplitter
from brain_decode_project.pl_modules import RegressionModule
from brain_decode_project.modules.budget_module import DummyBudgetManager
from brain_decode_project.networks.searchspaces import TCNSearchSpace
from braindecode.models.tcn import TCN

from pathlib import Path


class AgeBenchmark(ComposedBenchmark):
    config_space = ComposedConfigurationSpace(
        configuration_spaces=[
            RegressionModule.get_configuration_space(seed=0),
            TCNSearchSpace.get_configuration_space(seed=0)
        ]
    )
    budget_manager_type = DummyBudgetManager

    network_type = partial(
        TCN,
        n_outputs=1,
        add_log_softmax=False,
    )

    lightning_model_type = RegressionModule

    data_set_type = partial(
        TUHData,
        data_target_name='age',
        cut_off_first_k_seconds=60,
        n_max_minutes=3,
        sfreq=100,
        rng=0,
        train_or_eval='both',  # train, eval, both
        only_healthy=False,
        standardization_op='exponential_moving_demean',
    )

    data_set_splitter_type = partial(
        TUHDataSplitter,
        input_window_samples=1600
    )

    use_augmentations = False



if __name__ == '__main__':
    from brain_decode_project import DATA_PATH_LOCAL

    result_path = Path(
        '/home/philipp/Dokumente/Studium/Masterarbeit/'
        'BrainSignalDecodeProject/test_results'
    )
    age_b = AgeBenchmark(
        data_path=DATA_PATH_LOCAL,
        result_path=result_path
    )

    age_b.objective_function(
        configuration={
            'num_levels': 1,
            'num_channels': 16,
            'kernel_size': 8,
            'dropout': 0.01,

            'batch_size': 64,
        },
        fidelity={
            'num_epochs': 5,
        }
    )
