import logging
import torch

from typing import Dict, Union

import ConfigSpace as CS

from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from brain_decode_project.optimizers import Lookahead, Ranger
from brain_decode_project.pl_callbacks import SaveSnapshotCallback


log = logging.getLogger(__name__)


class BaseModule(LightningModule):
    def __init__(self, model, configuration: Dict, fidelity: Dict):
        super(BaseModule, self).__init__()

        # Fix an error that occurs because pl adds the model to the
        # configuration during reloading a checkpoint.
        if 'model' in configuration:
            configuration['model'] = None
        if 'fidelity' in configuration:
            del configuration['fidelity']

        from pprint import pprint
        pprint(configuration)

        self.save_hyperparameters(configuration, ignore=['model', 'fidelity', ])

        if fidelity is None:
            fidelity = dict(training_time_in_s=-1)
        self.fidelity = fidelity

        self.model = model
        self.min_learning_rate = 1e-5

        self.train_step_metrics = []
        self.all_metrics = []

    def forward(self, x):
        raise NotImplementedError()

    def configure_optimizers(self):

        if self.hparams.optimizer == "ExtendedAdam":
            optimizer = AdamW(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.optimizer == "ExtendedAdamLookAhead":
            optimizer = AdamW(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
            optimizer = Lookahead(
                optimizer=optimizer,
                k=self.hparams.lookahead_steps,
            )

        elif self.hparams.optimizer == "Ranger":

            optimizer = Ranger(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                k=self.hparams.lookahead_steps,
                weight_decay=self.hparams.weight_decay,
            )

        else:
            raise ValueError(
                f"Optimizer not parsable: {self.hparams.optimizer}. Must be either one of"
                "ExtendedAdam, ExtendedAdamLookAhead, Ranger."
            )

        # TODO: Add Learning Rate Scheduler to SearchSpace.
        #   Practical Issue: Large Variance in Training Time. Unclear how to set
        #   Restart threshold.
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.lr_scheduler_tmax,
            T_mult=1,
            eta_min=self.min_learning_rate,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def validation_epoch_end(self, outputs):
        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]

        for dataset_idx, output_per_dataloader in enumerate(outputs):
            dataset_name = 'valid' if dataset_idx == 0 else 'train'
            self._shared_evaluate_method(dataset_name, output_per_dataloader)

    def test_epoch_end(self, outputs):
        self._shared_evaluate_method(dataset_name='test', output_per_dataloader=outputs)

    def on_train_epoch_start(self) -> None:
        for metric in self.train_step_metrics:
            metric.reset()

    def _reset_metrics(self):
        for metric in self.all_metrics:
            metric.reset()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics()

    def on_test_epoch_start(self) -> None:
        self._reset_metrics()

    def _shared_evaluate_method(self, dataset_name, output_per_dataloader):
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameters():
        learning_rate = CS.UniformFloatHyperparameter(
            "learning_rate", lower=0.00001, upper=0.1, default_value=0.001, log=True
        )

        weight_decay = CS.UniformFloatHyperparameter(
            "weight_decay", lower=10 ** -9, upper=10 ** -3, default_value=10 ** -5, log=True
        )

        optimizer = CS.CategoricalHyperparameter(
            "optimizer", choices=["ExtendedAdam", "ExtendedAdamLookAhead", "Ranger"],
            default_value="ExtendedAdamLookAhead"
        )

        lr_scheduler_tmax = CS.UniformIntegerHyperparameter(
            "lr_scheduler_tmax", lower=-1, upper=1000, default_value=-1
        )

        lookahead_steps = CS.UniformIntegerHyperparameter(
            "lookahead_steps", lower=1, upper=10, default_value=1
        )

        lookahead_condition = CS.InCondition(
            child=lookahead_steps, parent=optimizer,
            values=["ExtendedAdamLookAhead", "Ranger"]
        )

        use_stochastic_weight_avg = CS.UniformIntegerHyperparameter(
            "use_stochastic_weight_avg", lower=0, upper=1, default_value=1,
        )

        hyperparameters = [
            learning_rate,
            weight_decay,
            optimizer,
            lookahead_steps,
            lr_scheduler_tmax,
            use_stochastic_weight_avg,
        ]

        conditions = [
            lookahead_condition
        ]
        return hyperparameters, conditions

    @staticmethod
    def get_configuration_space(
            seed: Union[int, None] = None,
    ) -> CS.ConfigurationSpace:

        hps, conds = BaseModule.get_hyperparameters()
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(hps)
        cs.add_conditions(conds)
        return cs

    def eval_using_snapshots(self, x):
        # Get the prediction with the "latest" model
        prediction_norm_list = [self.model(x).unsqueeze(0)]

        snapshot_callback = SaveSnapshotCallback.retrieve_cb_from_callbacks(
            self.trainer.callbacks
        )

        if snapshot_callback is not None and snapshot_callback.enable_snapshots:
            # Save the current weights.
            # We load them again after the prediction is not mix something here.
            snapshot_callback.store_current_model(self.model.state_dict())
            for snapshot_id in range(snapshot_callback.ensemble_size):
                snapshot_path = snapshot_callback.get_file_path(snapshot_id)
                if not snapshot_path.exists():
                    continue

                # load model weights
                state_dict = torch.load(snapshot_path)
                self.model.load_state_dict(state_dict['model_state_dict'])
                prediction_norm_list.append(self.model(x).unsqueeze(0))

            state_dict = snapshot_callback.reload_current_model_state()
            self.model.load_state_dict(state_dict)
            log.debug(f'Average Predictions from {len(prediction_norm_list)} Snapshots.')
        else:
            log.debug('Dont use snapshots!')

        prediction_norm = torch.concat(prediction_norm_list, dim=0)
        prediction_norm = torch.mean(prediction_norm, dim=0)

        return prediction_norm, prediction_norm_list[0].squeeze(dim=0)