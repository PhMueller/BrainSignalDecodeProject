from functools import partial

from brain_decode_project.benchmarks.composed_benchmark import ComposedBenchmark, \
    ComposedConfigurationSpace
from brain_decode_project.data import TUHData
from brain_decode_project.data.tuh_data import TUHDataSplitter
from brain_decode_project.pl_modules import RegressionModule
from brain_decode_project.pl_modules.baseline_module import BaselineRegressionPLModule
from brain_decode_project.modules.budget_module import DummyBudgetManager
from brain_decode_project.networks.searchspaces import TCNSearchSpace
from braindecode.models.tcn import TCN

from pathlib import Path


class AgeBaselineBenchmark(ComposedBenchmark):
    config_space = ComposedConfigurationSpace(
        configuration_spaces=[
            BaselineRegressionPLModule.get_configuration_space(),
            TCNSearchSpace.get_configuration_space(seed=0)
        ],
        remove_hp=['lookahead_steps', 'use_stochastic_weight_avg'],
        replace_hp_mapping={'optimizer': 'ExtendedAdam'}
    )


    budget_manager_type = DummyBudgetManager

    network_type = partial(
        TCN,
        n_outputs=1,
        add_log_softmax=False,
    )

    lightning_model_type = BaselineRegressionPLModule

    data_set_type = partial(
        TUHData,
        data_target_name='age',
        cut_off_first_k_seconds=2 * 60,
        n_max_minutes=5,
        sfreq=100,
        rng=0,
        train_or_eval='both',  # train, eval, both
        only_healthy=False,
        standardization_op='exponential_moving_standardize',
    )

    # Split the train data into 80% train and 20% validation.
    # or if called with objective_function_test: Use the entire train data set for training and
    # the extra test data set for testing.
    data_set_splitter_type = partial(
        TUHDataSplitter,
        input_window_samples=1600
    )


class AgeSmallSearchSpaceBenchmark(ComposedBenchmark):

    config_space = ComposedConfigurationSpace(
        configuration_spaces=[
            RegressionModule.get_configuration_space(seed=0),
            TCNSearchSpace.get_configuration_space(seed=0)
        ],
        replace_hp_mapping={
            'lr_scheduler_tmax': 1000,
            'optimizer': 'ExtendedAdam',

            'batch_size': 64,
            'kernel_size': 32,
            'num_channels': 32,
            'num_levels': 2,
        },
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
        cut_off_first_k_seconds=3 * 60,
        n_max_minutes=5 * 60,
        sfreq=100,
        rng=0,
        train_or_eval='both',  # train, eval, both
        only_healthy=False,
        standardization_op='exponential_moving_demean',
    )

    # Split the train data into 80% train and 20% validation.
    # or if called with objective_function_test: Use the entire train data set for training and
    # the extra test data set for testing.
    data_set_splitter_type = partial(
        TUHDataSplitter,
        input_window_samples=1600
    )

    use_augmentations = False


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

    # Split the train data into 80% train and 20% validation.
    # or if called with objective_function_test: Use the entire train data set for training and
    # the extra test data set for testing.
    data_set_splitter_type = partial(
        TUHDataSplitter,
        input_window_samples=1600
    )

    use_augmentations = False


if __name__ == '__main__':
    from brain_decode_project import DATA_PATH_LOCAL

    # result_path = Path(
    #     '/home/philipp/Dokumente/Studium/Masterarbeit/'
    #     'BrainSignalDecodeProject/test_results'
    # )
    # age_b = AgeBenchmark(
    #     data_path=DATA_PATH_LOCAL,
    #     result_path=result_path
    # )
    #
    # age_b.objective_function(
    #     configuration={
    #         'num_levels': 1,
    #         'num_channels': 16,
    #         'kernel_size': 8,
    #         'dropout': 0.01,
    #
    #         'batch_size': 64,
    #         'learning_rate': 0.01,
    #         'lr_scheduler_tmax': 614,
    #
    #         'optimizer': 'ExtendedAdam',
    #         'use_stochastic_weight_avg': 0,
    #         'weight_decay': 1.2111296908674393e-07
    #     },
    #     fidelity={
    #         'num_epochs': 5,
    #         'training_time_in_s': 500
    #     },
    #     n_recordings_to_load=300,
    # )

    result_path = Path(
        '/home/philipp/Dokumente/Studium/Masterarbeit/'
        'BrainSignalDecodeProject/test_results2'
    )
    age_b = AgeBaselineBenchmark(
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
            'learning_rate': 0.01,
            'lr_scheduler_tmax': 614,

            'optimizer': 'ExtendedAdam',
            # 'use_stochastic_weight_avg': 0,
            'weight_decay': 1.2111296908674393e-07,

            'use_augmentation': 1,
        },
        fidelity={
            'num_epochs': 7,
            'training_time_in_s': 500
        },
        n_recordings_to_load=300,
    )
