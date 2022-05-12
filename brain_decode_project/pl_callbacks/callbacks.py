import logging
import os
from pathlib import Path
from time import time
from typing import Optional, Any, Union, List, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

_logger = logging.getLogger(__name__)


class CountTrainingTimeCallBack(Callback):
    """
    This Callback Class handles taking time and stopping when a certain limit is reached.
    It can stop a run when either the `training_time_limit_in_s` or the `training_epoch_limit` is reached.

    """
    def __init__(self):

        self.start_time = time()
        self.train_epoch_start_time = time()
        self.train_epoch_end_time = time()
        self.train_batch_start_time = time()
        self.last_train_batch_start_time = time()
        self.train_batch_end_time = time()
        self.time_used_for_training = 0
        self.epochs_used_for_training = 0

        self.py_logger = logging.getLogger(__name__)

        super(CountTrainingTimeCallBack).__init__()

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.train_epoch_start_time = time()

    def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: Optional = None) \
            -> None:
        self.epochs_used_for_training += 1
        self.train_epoch_end_time = time()

        pl_module.log_dict({'epoch': float(self.epochs_used_for_training),
                            'start_time_epoch': self.train_epoch_start_time - self.start_time,
                            'finish_time_epoch': self.train_epoch_end_time - self.start_time,
                            'train_time_used': self.time_used_for_training},
                           on_epoch=True, logger=True, prog_bar=False)

    def on_train_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule',
                             batch: Any, batch_idx: int, unused: int = 0) -> None:
        # _logger.debug('COUNTER ON TRAIN BATCH START')

        self.last_train_batch_start_time = self.train_batch_start_time
        self.train_batch_start_time = time()

    # def on_after_backward(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    #     _logger.debug('COUNTER ON AFTER BACKWARD')

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: Any,
                           batch: Any, batch_idx: int, unused: int = 0) -> None:
        # _logger.debug('COUNTER ON TRAIN BATCH END')
        self.train_batch_end_time = time()
        self.time_used_for_training += time() - self.train_batch_start_time

    # def on_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    #     _logger.debug('COUNTER ON BATCH END')

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        pl_module.log_dict({'epoch': float(self.epochs_used_for_training),
                            'start_time_epoch': self.train_epoch_start_time - self.start_time,
                            'finish_time_epoch': self.train_epoch_end_time - self.start_time,
                            'train_time_used': self.time_used_for_training},
                           on_epoch=True, logger=True, prog_bar=False)

    @staticmethod
    def retrieve_callback_from_list(callbacks):
        for cb in callbacks:
            if isinstance(cb, CountTrainingTimeCallBack):
                return cb
        else:
            raise ValueError('We could not find a CountTrainingTimeCallBack. But this callback only works with it')

    def state_dict(self) -> Dict[str, Any]:
        state = dict(start_time=self.start_time,
                     train_epoch_start_time=self.train_epoch_start_time,
                     train_epoch_end_time=self.train_epoch_end_time,
                     # train_batch_start_time=self.train_batch_start_time,
                     time_used_for_training=self.time_used_for_training,
                     epochs_used_for_training=self.epochs_used_for_training,)
        _logger.debug('Count Training Time: Save checkpoint')
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.start_time = state_dict['start_time']
        self.train_epoch_start_time = state_dict['train_epoch_start_time']
        self.train_epoch_end_time = state_dict['train_epoch_end_time']
        # self.train_batch_start_time = state_dict['train_batch_start_time']
        self.time_used_for_training = state_dict['time_used_for_training']
        self.epochs_used_for_training = state_dict['epochs_used_for_training']
        _logger.debug(f'Count Training Time: Restored checkpoint {state_dict}')


class PrintCallback(Callback):

    def __init__(
            self, print_every_n_epochs: int = 50, print_every_n_steps: Optional[int] = None,
            print_validation_message: Optional[bool] = True,
    ):

        self.print_every_n_epochs = print_every_n_epochs
        self.print_every_n_steps = print_every_n_steps
        self.print_validation_message = print_validation_message
        self.last_epoch_printed = -1
        self.last_step_printed = -1

        self.py_logger = logging.getLogger('PrintCallback')

        super(PrintCallback, self).__init__()

    def __log_statement(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:

        self.py_logger.info(f'Current Epoch: {pl_module.current_epoch}|{trainer.max_epochs}  - '
                            f'Step: {pl_module.global_step}')

        for cb in trainer.callbacks:
            if isinstance(cb, CountTrainingTimeCallBack):
                self.py_logger.info(f'Current Time Limit: '
                                    f'{cb.time_used_for_training:.2f}|{pl_module.fidelity["training_time_in_s"]:.2f}'
                                    f' - (Total Time {time() - cb.start_time:.2f}s)')

                # Calculate the time, we waste outside the training.
                # This is the time between ending a batch and starting a new one.
                if self.py_logger.level == logging.DEBUG:
                    time_for_batch = cb.train_batch_end_time - cb.last_train_batch_start_time
                    time_for_step = cb.train_batch_start_time - cb.last_train_batch_start_time
                    self.py_logger.debug(f' - Time Batch: {time_for_batch:.4f} '
                                         f' - Time Step: {time_for_step:.4f}'
                                         f' - DIFF: {time_for_step - time_for_batch:4f}')

                break

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:

        self.py_logger.info('Training Start!')
        self.py_logger.info(f'Train:     {len(trainer.train_dataloader)} batches a '
                            f'{trainer.train_dataloader.loaders.batch_size} samples')
        self.py_logger.info(f'Valid:     {len(trainer.val_dataloaders[0])} batches a '
                            f'{trainer.val_dataloaders[0].batch_size} samples')

        # if len(trainer.val_dataloaders) > 1:
            # self.py_logger.info(f'Train Det: {len(trainer.val_dataloaders[0])} batches a '
            #                     f'{trainer.val_dataloaders[0].batch_size} samples')

        self.py_logger.info("Training is started!")

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:

        self.py_logger.info("Training is done.")

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:

        if (pl_module.current_epoch % self.print_every_n_epochs) == 0 \
                and self.last_epoch_printed != pl_module.current_epoch:

            self.__log_statement(trainer, pl_module)
            self.last_epoch_printed = pl_module.current_epoch

    def on_train_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule',
                             batch: Any, batch_idx: int,  unused: int = 0) -> None:

        if self.print_every_n_steps is not None \
                and (pl_module.global_step % self.print_every_n_steps) == 0 \
                and self.last_step_printed != pl_module.global_step:

            self.__log_statement(trainer, pl_module)
            self.last_epoch_printed = pl_module.current_epoch
            self.last_step_printed = pl_module.global_step

    def on_validation_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.print_validation_message:
            self.py_logger.info('Validation Start!')

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.print_validation_message:
            self.py_logger.info('Valdiation Finished!')

    def state_dict(self) -> Dict[str, Any]:
        state = dict(last_epoch_printed=self.last_epoch_printed,
                     last_step_printed=self.last_step_printed)
        self.py_logger.debug('Save checkpoint')
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.last_epoch_printed = state_dict['last_epoch_printed']
        self.last_step_printed = state_dict['last_step_printed']
        self.py_logger.debug(f'Restored checkpoint {state_dict}')


class StopWhenLimitIsReachedCallback(Callback):
    """
    This Callback Class handles taking time and stopping when a certain limit is reached.
    It can stop a run when either the `training_time_limit_in_s` or the `training_epoch_limit` is reached.

    """

    def __init__(
            self,
            training_time_limit_in_s: Optional[int] = None,
            training_epoch_limit: Optional[int] = None,
            validate_on_end: bool = True
    ):

        assert training_time_limit_in_s is not None or training_epoch_limit is not None, \
            'You have to specify either the time limit or the epoch limit, but both are None'

        self.py_logger = logging.getLogger('StopWhenLimitIsReachedCallback')

        self.training_time_limit_in_s = training_time_limit_in_s
        self.training_epoch_limit = training_epoch_limit
        self.validate_on_end = validate_on_end
        self.limit_reached = False

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.limit_reached:
            trainer.should_stop = True

    def on_train_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule',
                             batch: Any, batch_idx: int, unused: int = 0) -> Union[int, None]:
        # _logger.debug('STOPPER ON TRAIN BATCH START')

        if self.limit_reached:
            if self.validate_on_end:
                self.py_logger.info('Validate before stopping the training.')
                old_device = trainer.model.device
                trainer.validate(model=trainer.model, dataloaders=trainer.val_dataloaders, verbose=False)
                trainer.model.to(torch.device(old_device))
            trainer.should_stop = True

            return -1

        counter = CountTrainingTimeCallBack.retrieve_callback_from_list(trainer.callbacks)

        # To stop the training process early, we can return a -1.
        # Do this when the limits are reached.
        if self.training_time_limit_in_s is not None \
                and counter.time_used_for_training >= self.training_time_limit_in_s:
            self.py_logger.info(f'The time limit is reached. We stop the training. '
                                f'Time used for training: {counter.time_used_for_training:.2f} '
                                f'Limit: {self.training_time_limit_in_s:.2f}')
            self.limit_reached = True
            return -1

        if self.training_epoch_limit is not None \
                and counter.epochs_used_for_training >= self.training_epoch_limit:
            self.py_logger.info(f'The epochs limit is reached. We stop the training. '
                                f'Epochs used for training: {counter.epochs_used_for_training:.2f} '
                                f'Limit: {self.training_epoch_limit:.2f}')
            self.limit_reached = True
            return -1


class ValidateByTimeCallback(Callback):
    """
    Call the validation procedure every N minutes.
    """
    def __init__(self, validate_every_n_minutes: Union[int, float] = 10):
        self.py_logger = logging.getLogger('ValidateByTimeCallback')

        self.validate_every_n_minutes = validate_every_n_minutes
        self.validate_every_n_seconds = 60 * self.validate_every_n_minutes
        self.last_validation = 0

    def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: Any, batch: Any,
                           batch_idx: int, unused: int = 0) -> None:

        # def on_treain_batch_end(self,  trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        counter = CountTrainingTimeCallBack.retrieve_callback_from_list(trainer.callbacks)

        if counter.time_used_for_training >= self.last_validation + self.validate_every_n_seconds:
            self.py_logger.debug(f'Validate Model every {self.validate_every_n_seconds:.2f}s. '
                                 f'Current training time: {counter.time_used_for_training:.2f}s '
                                 f'Last validation was: {self.last_validation:.2f}s')
            old_device = trainer.model.device
            self.last_validation = counter.time_used_for_training
            trainer.validate(model=trainer.model, dataloaders=trainer.val_dataloaders, verbose=False)
            trainer.model.to(torch.device(old_device))

    def state_dict(self) -> Dict[str, Any]:
        state = dict(last_validation=self.last_validation)
        self.py_logger.debug('Save checkpoint')
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.last_validation = state_dict['last_validation']
        self.py_logger.debug(f'Restored checkpoint {state_dict}')


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.

    Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"

            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class SaveSnapshotCallback(Callback):

    def __init__(
            self,
            snapshot_dir: Union[str, Path],
            hashed_config_fidelity: str,
            enable_snapshots: bool = False
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.hashed_config_fidelity = hashed_config_fidelity
        self.enable_snapshots = enable_snapshots

        # We only want to save the latest 5 snapshots.
        # Store them in a cyclic manner. 1->2->..->5->1.
        self.snapshot_number = 0
        self.ensemble_size = 5
        self.py_logger = logging.getLogger('SaveSnapshotCallback')
        self.snapshot_dir.mkdir(exist_ok=True, parents=True)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:

        schedulers = pl_module.lr_schedulers()

        # Continue if no scheduler is used at all
        if schedulers is None:
            return

        # They return a list of schedulers if multiple are present. In our experiments,
        # we always use only one. However, it is better to be safe than sorry.
        scheduler = schedulers[0] if isinstance(schedulers, List) else schedulers

        # Snapshotting is only enabled for the Scheduler with Restarts:
        if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            return

        # Create a snap shot shortly beefore the restart
        if scheduler.T_cur != (scheduler.T_i - 1):
            return

        self.py_logger.info(f'Save snapshot #{self.snapshot_number}')
        torch.save(
            {
                'model_state_dict': pl_module.model.state_dict(),
                'epoch': trainer.current_epoch
            },
            self.snapshot_dir / f'{self.hashed_config_fidelity}_{self.snapshot_number}.ckpt'
        )

        # Increase the snapshot number
        self.snapshot_number = (self.snapshot_number + 1) % self.ensemble_size

    def state_dict(self) -> Dict[str, Any]:
        """
        Define how to store this callback in a checkpoint.
        """
        state = dict(
            snapshot_number=self.snapshot_number,
            snapshot_dir=self.snapshot_dir,
            enable_snapshots=self.enable_snapshots,
            hashed_config_fidelity=self.hashed_config_fidelity
        )
        self.py_logger.debug('Save checkpoint')
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Define how to reload the callback from a checkpoint.
        """
        self.snapshot_number = state_dict['snapshot_number']
        self.snapshot_dir = state_dict['snapshot_dir']
        self.enable_snapshots = state_dict['enable_snapshots']
        self.hashed_config_fidelity = state_dict['hashed_config_fidelity']

        self.snapshot_dir.mkdir(exist_ok=True, parents=True)

        self.py_logger.debug(f'Restored checkpoint {state_dict}')

    def get_file_path(self, snapshot_number: int) -> Path:
        """ Getter: Get path of the snapshot identified by the snapshot number. """
        return self.snapshot_dir / f'{self.hashed_config_fidelity}_{snapshot_number}.ckpt'

    def get_current_file_path(self) -> Path:
        """ Getter: Sometimes we need to store the current weights to a file to retrieve them later again.
            We use for this a constant file name.
        """
        return self.snapshot_dir / f'{self.hashed_config_fidelity}_current.ckpt'

    @staticmethod
    def retrieve_cb_from_callbacks(
            callbacks: List[Callback]
    ) -> Union[None, 'SaveSnapshotCallback']:

        """ Try to retrieve the snapshot callback from a list of (the trainer's) callbacks"""
        snapshot_callback = None
        for cb in callbacks:
            if isinstance(cb, SaveSnapshotCallback):
                snapshot_callback = cb
        return snapshot_callback

    def store_current_model(self, model_state_dict: Dict) -> None:
        """ Helperfunction: Save the weights of the current model to disk. """
        torch.save(
            {'model_state_dict': model_state_dict},
            str(self.get_current_file_path())
        )

    def reload_current_model_state(self) -> Dict:
        """ Helperfunction: Load the weights of the current model from disk. """
        state_dict = torch.load(self.get_current_file_path())
        return state_dict['model_state_dict']
