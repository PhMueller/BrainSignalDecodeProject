import logging
import torch

from abc import ABCMeta
from pathlib import Path
from typing import Union, Dict, List, Type, Any
from functools import partial
from time import time

import ConfigSpace as CS
import numpy as np
from hpobench.abstract_benchmark import AbstractBenchmark
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from torch.utils.data import DataLoader
from braindecode.models.util import get_output_shape

from brain_decode_project.data.base_data import BaseData, BaseDataSplitter
from brain_decode_project.modules.budget_module import BaseBudgetManager
from brain_decode_project.modules.trainer_module import BaseTrainer
from brain_decode_project.modules.result_module import ResultManager, HashTool

from brain_decode_project.pl_callbacks import StochasticWeightAveraging, ModelCheckpoint, \
    SaveSnapshotCallback, StopWhenLimitIsReachedCallback, PrintCallback, CountTrainingTimeCallBack

from brain_decode_project import SHOW_PROGRESSBAR, DEBUG_SETTINGS


class CustomMetaClass(ABCMeta):

    def __new__(cls, clsname, bases, attrs):

        # # ---------------------- TUH Data -------------------------------------------------------
        # # Check if the required parameters are given for the tuh data argument.
        # if (attrs.get('data_set_type') is not None
        #     and
        #     (
        #         # Some classes are pre-parameterized as partial functions.
        #         (
        #             isinstance(attrs.get('data_set_type'), partial)
        #             and issubclass(attrs.get('data_set_type').func, TUHData)
        #         )
        #         or
        #         # It is the tuh data class
        #         issubclass(attrs.get('data_set_type'), TUHData)
        #     )
        # ):
        #
        #     curr_target = attrs.get('target', None)
        #     assert curr_target is not None, \
        #         'MalConfigurationException: Wrongly configured benchmarks.' \
        #         'You are using a TUH Dataset, however you have not specified a `target`'
        #
        #     if isinstance(curr_target, str):
        #         curr_target = curr_target.lower()
        #         attrs['target'] = curr_target  # cast the target name to all lower chars.
        #
        #     available_targets = ['pathological', 'gender', 'age']
        #     assert curr_target in available_targets, \
        #         'MalConfigurationException: Wrongly configured benchmarks.' \
        #         f'You have specified an unknown dataset. ' \
        #         f'The given `target` was {curr_target} but has to be one of ' \
        #         f'{", ".join(available_targets)}'

        new_class = super().__new__(cls, clsname, bases, attrs)

        # We need to overwrite the get_configuration_space function to
        # link to the correct configuration space object.
        def get_configuration_space(seed: Union[int, None] = None) -> CS.configuration_space:
            composed_search_space: ComposedConfigurationSpace = attrs.get('config_space')
            config_space = composed_search_space.get_configuration_space(seed=seed)
            return config_space

        def get_fidelity_space(seed: Union[int, None] = None) -> CS.configuration_space:
            budget_manager: BaseBudgetManager = attrs.get('budget_manager_type')
            fidelity_space = budget_manager.get_fidelity_space()
            return fidelity_space

        def get_meta_information() -> Dict:

            def __try_get_name(attr) -> str:
                if isinstance(attr, partial):
                    name = attr.func.__name__
                else:
                    name = attr.__name__
                return name

            return {
                'Network': __try_get_name(attrs['network_type']),
                'Budget': __try_get_name(attrs['budget_manager_type']),
                'PL module': __try_get_name(attrs['lightning_model_type']),
                'Data set': __try_get_name(attrs['data_set_type']),
                'Trainer': __try_get_name(attrs['trainer_type']),
            }

        setattr(new_class, 'get_configuration_space', staticmethod(get_configuration_space))
        setattr(new_class, 'get_fidelity_space', staticmethod(get_fidelity_space))
        setattr(new_class, 'get_meta_information', staticmethod(get_meta_information))

        return new_class


class ComposedConfigurationSpace:
    logger = logging.getLogger('ComposedConfigurationSpace')

    def __init__(
            self,
            configuration_spaces: List[CS.ConfigurationSpace],
            remove_hp: Union[List[str], None] = None,
            replace_hp_mapping: Union[Dict[str, Any], None] = None,
            seed: Union[int, None] = None
    ):
        """
        This class combines multipe configuration spaces.
        It also allows to remove or replace some parameters within the space.

        Parameters
        ----------
        configuration_spaces: List[CS.ConfigurationSpace]
            A collection of Search Spaces to combine

        remove_hp: List[str]
            A list of hyperparameters that are removed from the search space.

        replace_hp_mapping: Dict[Str]
            This dictionary defines a mapping of parameter that should be replaced with some
            constant values

            It has the form:
            { 'Old HP to replace (Name)': 'New HP', ...}

        seed: Union[int, None] = None
            Random Seed
        """

        self.configuration_space = ComposedConfigurationSpace.combine_configuration_spaces(
            configuration_spaces=configuration_spaces,
            seed=seed
        )

        self.configuration_space = ComposedConfigurationSpace.remove_hp_fom_cs(
            old_cs=self.configuration_space,
            remove_hp=remove_hp,
            replace_hp_mapping=replace_hp_mapping,
            seed=seed
        )

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Getter function """
        if seed is not None:
            self.configuration_space.seed(seed)
        return self.configuration_space

    @staticmethod
    def combine_configuration_spaces(
            configuration_spaces: List[CS.ConfigurationSpace],
            seed: Union[int, None] = None
    ) -> CS.ConfigurationSpace:

        composed_cs = CS.ConfigurationSpace(seed=seed)

        for cs in configuration_spaces:
            composed_cs.add_configuration_space(configuration_space=cs, prefix='', delimiter='')

        return composed_cs

    @staticmethod
    def remove_hp_fom_cs(
            old_cs: CS.ConfigurationSpace,
            remove_hp: Union[List[str], None] = None,
            replace_hp_mapping: Union[Dict[str, Any], None] = None,
            seed: Union[int, None] = None,
    ) -> CS.ConfigurationSpace:

        new_cs = CS.ConfigurationSpace(seed=seed)

        # ---------------------------- Remove hyperparameters ------------------------------------
        if remove_hp is None:  # No HP given to remove
            remove_hp = []

        # We also remove the hp that we later replace
        if replace_hp_mapping is not None and len(replace_hp_mapping) != 0:
            remove_hp += list(replace_hp_mapping.keys())

        new_cs.add_hyperparameters(
            [hp for hp in old_cs.get_hyperparameters() if hp.name not in remove_hp]
        )

        # ---------------------------- Replace hyperparameters -----------------------------------
        constant_hps = {}
        if replace_hp_mapping is not None and len(replace_hp_mapping) != 0:
            constant_hps = {
                hp_name: CS.Constant(hp_name, hp_value)
                for hp_name, hp_value in replace_hp_mapping.items()
            }
            new_cs.add_hyperparameters(constant_hps.values())

        # ---------------------------- Conditions ------------------------------------------------
        # Only add conditions that still have their target/child:
        for condition in old_cs.get_conditions():  # However they also need their parent.
            if condition.child.name in new_cs.get_hyperparameter_names():
                if condition.parent.name in remove_hp:
                    condition.parent = constant_hps[condition.parent.name]
                if condition.child.name in remove_hp:
                    condition.child = constant_hps[condition.child.name]
                try:
                    new_cs.add_condition(condition)
                except ValueError as e:
                    ComposedConfigurationSpace.logger.warning(
                        f'We were not able to add the following condition: '
                        f'{condition} due to Exception: {e}'
                    )
        return new_cs

    def __repr__(self):
        return self.get_configuration_space().__repr__()


class ComposedBenchmark(AbstractBenchmark, metaclass=CustomMetaClass):

    config_space: CS.ConfigurationSpace = None
    budget_manager_type: Type[BaseBudgetManager] = None

    network_type = None
    lightning_model_type = None

    data_set_type: Union[Type[BaseData], partial] = None
    data_set_splitter_type: Union[Type[BaseDataSplitter], partial] = None
    # trainer_type: BaseTrainer = None
    use_augmentations: Union[bool, None] = False

    def __init__(
            self,
            data_path: Path,
            result_path: Path,
            rng: Union[int, np.random.RandomState, None] = None,
            **kwargs
    ):

        assert self.config_space is not None
        assert self.budget_manager_type is not None
        assert self.network_type is not None
        assert self.lightning_model_type is not None
        assert self.data_set_type is not None
        assert self.data_set_splitter_type is not None
        # assert self.data_manager_type is not None
        # assert self.trainer_type is not None

        super(ComposedBenchmark, self).__init__(rng=rng, **kwargs)  # TODO: implement rng

        self.result_path = Path(result_path)
        self.result_path.mkdir(exist_ok=True, parents=True)

        self.data_path = Path(data_path)

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return ComposedBenchmark.config_space

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        return ComposedBenchmark.budget_manager_type.get_fidelity_space()

    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[Dict, CS.Configuration, None] = None,
            custom_training_time_in_s: Union[int, None] = None,
            custom_training_epoch_limit: Union[int, None] = None,
            rng: Union[np.random.RandomState, int, None] = None,
            use_final_eval: Union[bool, None] = False,
            disable_checkpoints: bool = True,
            load_model: bool = True,
            custom_checkpoint_dir: Union[None, Path] = None,
            debug: bool = False,
            n_recordings_to_load: Union[int, None] = None,
            **kwargs
    ) -> Dict:

        self.logger.info(
            f'Objective Function called with: '
            f'n_recordings_to_load: {n_recordings_to_load}; debug: {debug};'
            f'use_final_eval: {use_final_eval}; disable_checkpoints: {disable_checkpoints}; '
            f'custom_checkpoint_dir: {custom_checkpoint_dir}'
        )

        configuration = AbstractBenchmark._check_and_cast_configuration(
            configuration, self.get_configuration_space()
        )
        configuration = configuration.get_dictionary()

        use_cuda = torch.cuda.is_available()
        seed_everything(seed=rng)

        # Use the Budget Manger to map from hb_budget to (num_epochs, training_time_in_s)
        budget_manager = self.budget_manager_type()
        fidelity = budget_manager.get_fidelity_from_hb_budget(fidelity)

        if custom_training_time_in_s is not None:
            self.logger.info(f'Set custom training time limit: {custom_training_time_in_s}')
            fidelity['training_time_in_s'] = custom_training_time_in_s
        if custom_training_epoch_limit is not None:
            self.logger.info(f'Set custom training epoch limit: {custom_training_epoch_limit}')
            fidelity['num_epochs'] = custom_training_epoch_limit

        result_manager = ResultManager(self.result_path)
        result_manager.update_mapping(configuration, fidelity)
        run_result_path = result_manager.get_run_result_directory(configuration, fidelity)
        self.logger.info(f'Automatically determined save directory: {run_result_path}')

        # Create the data set. This might take a while. This step solely contains loading the
        # braindecode data set
        dataset = self.data_set_type(
            data_path=self.data_path,
            n_recordings_to_load=n_recordings_to_load,
        )

        model = self.network_type(
            dataset.num_channels,
            # TODO: Write a mapping from config to network
            n_blocks=configuration['num_levels'],
            n_filters=configuration['num_channels'],
            kernel_size=configuration['kernel_size'],
            drop_prob=configuration['dropout'],
        )

        output_shape = get_output_shape(
            model, dataset.num_channels,
            self.data_set_splitter_type.keywords['input_window_samples']
        )
        n_preds_per_input = output_shape[2]
        
        # ---------------------------- INIT DATA SET -------------------------------------------------------------------
        dataset_splitter = self.data_set_splitter_type(window_stride_samples=n_preds_per_input)
        train_set, valid_set = dataset_splitter.split_into_train_valid(dataset)

        train_dl = DataLoader(train_set, batch_size=configuration['batch_size'], drop_last=True)
        valid_dl = DataLoader(valid_set, batch_size=configuration['batch_size'], drop_last=False)
        
        # ---------------------------- INIT MODEL ----------------------------------------------------------------------
        # Set automatically the length for the LR restart cycle to the maximum number of epochs.
        if configuration.get('lr_scheduler_tmax', -1) == -1:
            configuration['lr_scheduler_tmax'] = fidelity['num_epochs']

        pl_module = self.lightning_model_type(
            model=model,
            configuration=configuration,
            fidelity=fidelity,
            y_mean=dataset.y_mean,
            y_std=dataset.y_std,
        )

        callbacks = [
            CountTrainingTimeCallBack(),
            StopWhenLimitIsReachedCallback(
                training_time_limit_in_s=fidelity['training_time_in_s'],
                training_epoch_limit=fidelity['num_epochs'],
                validate_on_end=False
            ),
            # EarlyStopping(
            #     monitor='valid_accuracy', mode='max', patience=10
            # ),
        ]

        if not SHOW_PROGRESSBAR:
            callbacks.append(PrintCallback(
                print_validation_message=False,
                print_every_n_epochs=10 if not debug else DEBUG_SETTINGS['print_epoch'],
                print_every_n_steps=None if not debug else DEBUG_SETTINGS['print_step']
            ))

        # Disable SWA for the regression task. -> Deepcopy error. TODO.
        if False and configuration.get('use_stochastic_weight_avg', False):
            callbacks.append(
                StochasticWeightAveraging(
                    swa_epoch_start=0.8,
                    annealing_epochs=int(0.2 * fidelity['num_epochs']),
                    device=None
                )
            )
        
        # ---------------------------- Checkpoint and Restarting -------------------------------------------------------
        # Step 1: Manually. Check that the specified custom checkpoint is a path object and links to an existing 
        #         ckpoint. 
        assert custom_checkpoint_dir is None \
            or (isinstance(custom_checkpoint_dir, Path) and custom_checkpoint_dir.is_file()), \
            'The checkpoint (if given) has to point to a .ckpt file but was ' \
            f'{custom_checkpoint_dir}'

        # Step 2: Automatically. If there exists a checkpoint at (config, fidelity) and no custom file is given, 
        #         use last saved checkpoint.
        last_checkpoint_file = run_result_path / 'checkpoints' / 'last.ckpt'
        if not disable_checkpoints \
                and custom_checkpoint_dir is None \
                and last_checkpoint_file.exists():
            self.logger.info(f'Start fom last checkpoint {last_checkpoint_file}')
            custom_checkpoint_dir = last_checkpoint_file
        
        # Add the model checkpoint callback. Observe the epoch-wise train metric. Save the model 
        # every time a validation was performed. 
        # This corresponds to the `check_val_every_n_epochs` of the trainer. 
        if not disable_checkpoints:
            if hasattr(pl_module, 'train_mse_metric'):
                monitor_metric = 'train_mse'
                monitor_mode = 'min'
            elif hasattr(pl_module, 'train_acc_metric'):
                monitor_metric = 'train_acc'
                monitor_mode = 'max'
            else:
                raise ValueError('Unknown pl module. Unclear which metric to montior.')
            
            callbacks.append(
                ModelCheckpoint(
                    monitor=monitor_metric,
                    filename="best" if not use_final_eval else "TrainTest_best",
                    dirpath=run_result_path / 'checkpoints',
                    mode=monitor_mode,
                    save_last=True,
                    save_top_k=2,
                )
            )
            
            # Add a callback for managing the snapshots
            if self.use_augmentations:
                callbacks.append(
                    SaveSnapshotCallback(
                        snapshot_dir=run_result_path / 'checkpoints' / 'snapshots',
                        hashed_config_fidelity=HashTool.create_hash_name(configuration, fidelity),
                        enable_snapshots=True,
                    )
                )

        # ---------------------------- INIT TRAINER --------------------------------------------------------------------
        trainer = Trainer(
            gpus=1 if use_cuda else 0,
            benchmark=True,
            deterministic=True,
            max_epochs=fidelity['num_epochs'],

            check_val_every_n_epoch=5,
            enable_checkpointing=not disable_checkpoints,
            default_root_dir=str(run_result_path),
            callbacks=callbacks,
            logger=[
                TensorBoardLogger(
                    save_dir=str(run_result_path / 'tb_logs'),
                    name=model.__class__.__name__,
                ),
                CSVLogger(
                    save_dir=str(run_result_path / 'csv_logs'),
                    name=model.__class__.__name__,
                    flush_logs_every_n_steps=10,
                )
            ],
            gradient_clip_val=0.5,
            gradient_clip_algorithm='norm',
            enable_progress_bar=SHOW_PROGRESSBAR,
        )

        # Problems with the stochastic Weight Averaging:
        # SWA does not support saving / loading from checkpoint -> no loading of avg model
        # Make sure to start swa in those cases again.
        swa_callback = [cb for cb in trainer.callbacks
                        if isinstance(cb, StochasticWeightAveraging)]

        if custom_checkpoint_dir is not None and len(swa_callback) != 0:
            swa_callback = swa_callback[0]
            old_state = torch.load(custom_checkpoint_dir)
            swa_start_epoch = swa_callback._swa_epoch_start
            if isinstance(swa_start_epoch, float):
                swa_start_epoch = int(trainer.max_epochs * swa_start_epoch)
            swa_callback._swa_epoch_start = max(old_state['epoch'], swa_start_epoch) + 1

        # ---------------------------- RUN TRAINING --------------------------------------------------------------------
        start_time = time()

        trainer.fit(
            model=pl_module,
            train_dataloaders=train_dl,
            val_dataloaders=[valid_dl, train_dl],
            ckpt_path=custom_checkpoint_dir if load_model else None,
        )

        # ---------------------------- RUN VALIDATION STEP -------------------------------------------------------------
        if not use_final_eval:
            results = trainer.validate(
                model=pl_module,
                dataloaders=[valid_dl, train_dl]
            )
        else:
            results = trainer.test(
                model=pl_module,
                dataloaders=valid_dl
            )

        costs = time() - start_time
        result_manager.register_intermediate_result(results, index=fidelity['i_cv_fold'])

        final_result = result_manager.register_final_result(
            configuration,
            fidelity,
            costs=costs,
            exception_str='',
            hb_budget=fidelity['num_epochs'],
            test_metric=use_final_eval,
        )
        return final_result

    def objective_function_test(
        self,
        configuration: Union[CS.Configuration, Dict],
        fidelity: Union[Dict, CS.Configuration, None] = None,
        custom_training_time_in_s: Union[int, None] = None,
        custom_training_epoch_limit: Union[int, None] = None,
        rng: Union[np.random.RandomState, int, None] = None,
        use_final_eval: Union[bool, None] = False,
        disable_checkpoints: bool = True,
        load_model: bool = True,
        custom_checkpoint_dir: Union[None, Path] = None,
        debug: bool = False,
        n_recordings_to_load: Union[int, None] = None,
        ** kwargs
    ) -> Dict:

        return self.objective_function(
            configuration=configuration, fidelity=fidelity,
            custom_training_time_in_s=custom_training_time_in_s,
            custom_training_epoch_limit=custom_training_epoch_limit,
            rng=rng,
            use_final_eval=True,
            disable_checkpoints=disable_checkpoints,
            custom_checkpoint_dir=custom_checkpoint_dir,
            debug=debug, n_recordings_to_load=n_recordings_to_load
        )
