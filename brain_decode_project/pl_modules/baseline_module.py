import logging
from typing import Dict, Union

import torch
from brain_decode_project.pl_modules.regression_module import PLModule as RegressionBase
import ConfigSpace as CS


log = logging.getLogger(__name__)


class BaselineRegressionPLModule(RegressionBase):
    """ This class presents the baseline following the experimental setup of a previous work. """
    def __init__(self, model, configuration: Dict, fidelity: Dict, y_mean, y_std):

        super(BaselineRegressionPLModule, self).__init__(model, configuration, fidelity, y_mean, y_std)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y, i = batch
        # ----------- Data Augmentation ----------------------------------------------------------
        # Patryk augmented his data with "scaling" and "flipping in time".
        # As far as I understood, he decided during the loading of each sample whether it should
        # be augmented or not.
        # I deviate from his approach by changing this to individually augmenting.
        # Also, I reduce the chance of applying "flip" from 50 to 20 percent, and the probability
        # for "scaling" from 100% to 20%
        if self.hparams['use_augmentation']:
            apply_scaling = torch.rand(len(x)) <= 0.2
            if torch.any(apply_scaling):
                # Scale the input data with values between 0.5 and 2.
                scale_factor = 2 ** torch.FloatTensor(x.shape[0]).uniform_(-1, 1)
                scale_factor = scale_factor.reshape((-1, 1, 1))
                x[apply_scaling] *= scale_factor[apply_scaling]

            apply_flip = torch.rand(len(x)) <= 0.2
            if torch.any(apply_flip):
                # Flip the input samples along the time axis
                x[apply_flip] = torch.flip(x[apply_flip], [2])

        return super(BaselineRegressionPLModule, self).training_step((x, y, i), batch_idx)

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        x, y, i = batch

        y_norm = (y - self.y_mean) / self.y_std

        prediction_norm = self.model(x)

        if dataset_idx is None or dataset_idx == 0:
            mse_metric, mse_metric_model = self.valid_mse_metric, self.valid_mse_metric_model
        else:
            mse_metric, mse_metric_model = self.train_mse_metric, self.train_mse_metric_model

        mse_loss = mse_metric(torch.mean(prediction_norm, dim=(1, 2)), y_norm)

        return {
            'mse_loss': mse_loss.item(),
            'prediction_norm': prediction_norm,
            'indices': i,
            'targets_norm': y_norm
        }

    def test_step(self, batch, batch_idx, dataset_idx=None):
        x, y, i = batch

        y_norm = (y - self.y_mean) / self.y_std
        prediction_norm = self.model(x)

        mse_loss = self.test_mse_metric(torch.mean(prediction_norm, dim=(1, 2)), y_norm)

        return {
            'mse_loss': mse_loss.item(),
            'prediction_norm': prediction_norm,
            'indices': i,
            'targets_norm': y_norm
        }

    @staticmethod
    def get_hyperparameters():
        hps = [
            CS.UniformIntegerHyperparameter(
                'use_augmentation', lower=0, upper=1, default_value=1, log=False,
            ),
            # That was not included in Patryks Search Space. The augmentation is performed a single time.
            # CS.UniformFloatHyperparameter(
            #     'prob_gaussian_noise', lower=0.0, upper=0.5, default_value=0.0, log=True
            # ),
            # CS.UniformFloatHyperparameter(
            #     'prob_flip', lower=0.0, upper=0.5, default_value=0.0, log=True
            # )
        ]
        return hps, []

    @staticmethod
    def get_configuration_space(
            seed: Union[int, None] = None,
    ) -> CS.ConfigurationSpace:

        cs = CS.ConfigurationSpace(seed=seed)
        base_hps, base_conds = RegressionBase.get_hyperparameters()
        cs.add_hyperparameters(base_hps)
        cs.add_conditions(base_conds)

        baseline_hps, _ = BaselineRegressionPLModule.get_hyperparameters()
        cs.add_hyperparameters(baseline_hps)
        return cs
