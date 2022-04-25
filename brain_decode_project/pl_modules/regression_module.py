import logging
from typing import Dict, Union

import ConfigSpace as CS
import numpy as np
import torch
import torchmetrics
from braindecode.training import trial_preds_from_window_preds

from brain_decode_project.pl_modules import BaseModule


log = logging.getLogger(__name__)


class PLModule(BaseModule):
    def __init__(self, model, configuration: Dict, fidelity: Dict, y_mean, y_std):
        super(PLModule, self).__init__(model, configuration, fidelity)

        self.y_mean = y_mean
        self.y_std = y_std

        self.train_step_mse_metric = torchmetrics.MeanSquaredError()
        self.train_mse_metric = torchmetrics.MeanSquaredError()
        self.train_mse_metric_model = torchmetrics.MeanSquaredError()
        self.valid_mse_metric = torchmetrics.MeanSquaredError()
        self.valid_mse_metric_model = torchmetrics.MeanSquaredError()
        self.test_mse_metric = torchmetrics.MeanSquaredError()
        self.test_mse_metric_model = torchmetrics.MeanSquaredError()

        self.train_step_metrics.append(self.train_step_mse_metric)
        self.all_metrics.extend([
            # self.train_step_mse_metric,
            self.train_mse_metric, self.train_mse_metric_model,
            self.valid_mse_metric, self.valid_mse_metric_model,
            self.test_mse_metric, self.test_mse_metric_model
        ])

    def forward(self, x):
        return self.model.forward(x).clip(0, 100)

    def training_step(self, batch, batch_idx):
        x, y, i = batch

        y = (y - self.y_mean) / self.y_std
        prediction = self.model(x)

        # Compute the MSE of this particular batch.
        mse = self.train_step_mse_metric(torch.mean(prediction, dim=(1, 2)), y)

        self.log(
            'train_mse_step', mse.item(),
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return mse

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        x, y, i = batch

        y_norm = (y - self.y_mean) / self.y_std

        prediction_norm, prediction_norm_model = self.eval_using_snapshots(x)

        if dataset_idx is None or dataset_idx == 0:
            mse_metric, mse_metric_model = self.valid_mse_metric, self.valid_mse_metric_model
        else:
            mse_metric, mse_metric_model = self.train_mse_metric, self.train_mse_metric_model

        mse_loss = mse_metric(torch.mean(prediction_norm, dim=(1, 2)), y_norm)
        mse_loss_without_snapshot = mse_metric_model(
            torch.mean(prediction_norm_model, dim=(1, 2)), y_norm
        )

        return {
            'mse_loss': mse_loss.item(),
            'mse_loss_without_snapshot': mse_loss_without_snapshot.item(),

            'prediction_norm': prediction_norm,
            'prediction_norm_model': prediction_norm_model,

            'indices': i,
            'targets_norm': y_norm
        }

    def test_step(self, batch, batch_idx, dataset_idx=None):
        x, y, i = batch

        y_norm = (y - self.y_mean) / self.y_std
        prediction_norm, prediction_norm_model = self.eval_using_snapshots(x)

        mse_loss = self.test_mse_metric(torch.mean(prediction_norm, dim=(1, 2)), y_norm)
        mse_loss_without_snapshot = self.test_mse_metric_model(
            torch.mean(prediction_norm_model, dim=(1, 2)), y_norm
        )

        return {
            'mse_loss': mse_loss.item(),
            'mse_loss_without_snapshot': mse_loss_without_snapshot.item(),
            'prediction_norm': prediction_norm,
            'prediction_norm_model': prediction_norm_model,
            'indices': i,
            'targets_norm': y_norm
        }

    def training_epoch_end(self, outputs):
        # Compute the mse across the predictions of the all batches of this epoch!
        self.log(
            'train_mse', self.train_step_mse_metric.compute().item(),
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.train_step_mse_metric.reset()

    def _shared_evaluate_method(self, dataset_name, output_per_dataloader):
        # Collect all predictions and combine them.
        all_preds, all_preds_model, all_is, all_ys = [], [], [], []
        for entry in output_per_dataloader:
            all_preds.extend(entry['prediction_norm'].cpu().numpy())

            if 'prediction_norm_model' in entry:
                all_preds_model.extend(entry['prediction_norm_model'].cpu().numpy())

            all_is.extend([list_entry.cpu() for list_entry in entry['indices']])
            all_ys.extend(entry['targets_norm'].cpu().numpy().astype(np.float32))

        all_preds = np.array(all_preds)
        all_ys = np.array(all_ys)

        # The Predictions returned by the model are normalized using the mean and variance.
        # Limit the normalized values to be in a good range [norm(0), norm(100)]
        min_value = (0 - self.y_mean) / self.y_std
        max_value = (100 - self.y_mean) / self.y_std
        all_preds = all_preds.clip(min_value, max_value)

        # Compute the cropwise metrics
        preds_cropwise = np.mean(all_preds, axis=(1, 2))  # TODO: Check this step again.
        mse_cropwise = np.mean((preds_cropwise - all_ys) ** 2)
        mae_cropwise_unnormed = np.mean(np.abs(self.y_std * (preds_cropwise - all_ys)))
        rmse_cropwise_unnormed = np.sqrt(mse_cropwise * (self.y_std ** 2))

        # Compute the trialwise metrics
        trial_ys = all_ys[np.diff(torch.cat(all_is[0::3]), prepend=[np.inf]) != 1]
        # noinspection PyTypeChecker
        preds_per_trial = trial_preds_from_window_preds(
            all_preds, torch.cat(all_is[0::3]), torch.cat(all_is[2::3])
        )

        preds_trialwise = np.array([p.mean() for p in preds_per_trial])
        mse_trialwise = np.mean((preds_trialwise - trial_ys) ** 2)
        mae_trialwise_unnormed = np.mean(np.abs(self.y_std * (preds_trialwise - trial_ys)))
        rmse_trialwise_unnormed = np.sqrt(mse_trialwise * (self.y_std ** 2))

        # Compute the trialwise metrics without the impact of snapshot ensembling.
        if len(all_preds_model) != 0:
            all_preds_model = np.array(all_preds_model)
            all_preds_model = all_preds_model.clip(min_value, max_value)

            # Concatenate all the crops per patient to a get the prediction for the entire trial
            preds_model_per_trial = trial_preds_from_window_preds(
                all_preds_model, torch.cat(all_is[0::3]), torch.cat(all_is[2::3])
            )

            preds_model_trialwise = np.array([p.mean() for p in preds_model_per_trial])
            mse_model_trialwise = np.mean((preds_model_trialwise - trial_ys) ** 2)
            mae_model_trialwise_unnormed = \
                np.mean(np.abs(self.y_std * (preds_model_trialwise - trial_ys)))
            rmse_model_trialwise_unnormed = np.sqrt(mse_model_trialwise * (self.y_std ** 2))
        else:
            mse_model_trialwise = torch.IntTensor([-1234])
            mae_model_trialwise_unnormed = torch.IntTensor([-1234])
            rmse_model_trialwise_unnormed = torch.IntTensor([-1234])

        result_dict = {
            f'{dataset_name}_function_value': mse_cropwise.item(),
            f'{dataset_name}_mse_cropwise': mse_cropwise.item(),
            f'{dataset_name}_mae_cropwise_unnormed': mae_cropwise_unnormed.item(),
            f'{dataset_name}_rmse_cropwise_unnormed': rmse_cropwise_unnormed.item(),

            f'{dataset_name}_mse_trialwise': mse_cropwise.item(),
            f'{dataset_name}_mae_trialwise_unnormed': mae_trialwise_unnormed.item(),
            f'{dataset_name}_rmse_trialwise_unnormed': rmse_trialwise_unnormed.item(),

            f'{dataset_name}_mse_model_trialwise': mse_model_trialwise.item(),
            f'{dataset_name}_mae_model_trialwise_unnormed': mae_model_trialwise_unnormed.item(),
            f'{dataset_name}_rmse_model_trialwise_unnormed': rmse_model_trialwise_unnormed.item(),

            f'{dataset_name}_num_samples': len(all_ys),
        }

        # Sanity Check. Report the accumulated metric from the step - function.
        if dataset_name == 'train':
            mse_cropwise_metric = self.train_mse_metric.compute().item()
        elif dataset_name == 'valid':
            mse_cropwise_metric = self.valid_mse_metric.compute().item()
        elif dataset_name == 'test':
            mse_cropwise_metric = self.test_mse_metric.compute().item()
        else:
            raise ValueError(
                f'Sanity Check. Dataset name is unknown. Has to be one of '
                f'[train, valid, test], but was {dataset_name}'
            )
        result_dict[f'{dataset_name}_mse_metric_cropwise'] = mse_cropwise_metric

        snapshot_used = 'mse_loss_without_snapshot' in output_per_dataloader[0]
        if snapshot_used:
            if dataset_name == 'train':
                mse_cropwise_metric_model = self.train_mse_metric_model.compute().item()
            elif dataset_name == 'valid':
                mse_cropwise_metric_model = self.valid_mse_metric_model.compute().item()
            elif dataset_name == 'test':
                mse_cropwise_metric_model = self.test_mse_metric_model.compute().item()
            else:
                raise ValueError(
                    f'Sanity Check. Dataset name is unknown. Has to be one of '
                    f'[train, valid, test], but was {dataset_name}'
                )
            result_dict[f'{dataset_name}_mse_metric_cropwise_wo_snapshot'] = \
                mse_cropwise_metric_model

        self.log_dict(
            result_dict,
            on_step=False, on_epoch=True, prog_bar=False, logger=True, add_dataloader_idx=False,
        )

        return result_dict


class AugmentationsPLModule(PLModule):

    def __init__(self, model, configuration: Dict, fidelity: Dict, y_mean, y_std):
        super(AugmentationsPLModule, self).__init__(
            model=model,
            configuration=configuration,
            fidelity=fidelity,
            y_mean=y_mean,
            y_std=y_std,
        )

        from brain_decode_project.networks.augementation_block import AugmentationBox
        from brain_decode_project.data.tuh_data import TUHData

        self.augmentation_box = AugmentationBox(
            p_time_reverse=configuration.get('p_time_reverse', 0),
            p_sign_flip=configuration.get('p_sign_flip', 0),
            p_ft_surrogate=configuration.get('p_ft_surrogate', 0),
            p_channel_shuffle=configuration.get('p_channel_shuffle', 0),
            p_channel_dropout=configuration.get('p_channel_dropout', 0),
            p_gaussian_noise=configuration.get('p_gaussian_noise', 0),
            p_channel_symmetry=configuration.get('p_channel_symmetry', 0),
            p_smooth_time_mask=configuration.get('p_smooth_time_mask', 0),
            # p_mixup=configuration.get('p_mixup', 0),  # Mixup is only for classification tasks.
            p_frequency_shift=configuration.get('p_frequency_shift', 0),
            # p_bandstop_filter=configuration.get('p_bandstop_filter', 0),  # Not relevant.
            p_sensor_rotation=configuration.get('p_sensor_rotation', 0),
            seed=0,
            sfreq=100,
            ordered_channel_names=TUHData.get_ordered_channel_names()
        )

    def training_step(self, batch, batch_idx):
        x, y, i = batch

        # Augment the batch. I normalize the batch, because the augemntations are calibrated on
        # for the normalized values.
        y = (y - self.y_mean) / self.y_std
        x, y = self.augmentation_box(x, y)
        y = (y * self.y_std) + self.y_mean

        return super(AugmentationsPLModule, self).training_step((x, y, i), batch_idx)

    # noinspection PyMethodOverriding
    @staticmethod
    def get_hyperparameters(add_gates):

        from brain_decode_project.networks.augementation_block import AugmentationBox
        hp_names = AugmentationBox.get_hp_names()

        hyperparameters = []
        conditions = []

        for hp_name in hp_names:
            hp = CS.UniformFloatHyperparameter(
                hp_name, lower=0.0, upper=0.5, default_value=0.25, log=False
            )
            hyperparameters.append(hp)

            if add_gates:
                gate = CS.UniformIntegerHyperparameter(
                    'use_' + hp_name, lower=0, upper=1, default_value=1
                )
                condition = CS.EqualsCondition(child=hp, parent=gate, value=1)
                hyperparameters.append(gate)
                conditions.append(condition)

        hyperparameters.append(
            CS.UniformIntegerHyperparameter(
                'use_label_smoothing', lower=0, upper=1, default_value=1
            )
        )

        return hyperparameters, conditions

    @staticmethod
    def get_configuration_space(
            seed: Union[int, None] = None,
            add_gates: bool = True
    ) -> CS.ConfigurationSpace:

        cs = CS.ConfigurationSpace(seed=seed)

        base_hps, base_conds = PLModule.get_hyperparameters()
        cs.add_hyperparameters(base_hps)
        cs.add_conditions(base_conds)

        augm_hps, augm_conds = AugmentationsPLModule.get_hyperparameters(add_gates=add_gates)
        cs.add_hyperparameters(augm_hps)
        cs.add_conditions(augm_conds)

        return cs

