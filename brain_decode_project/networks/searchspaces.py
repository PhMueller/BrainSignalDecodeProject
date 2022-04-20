from typing import Union
import ConfigSpace as CS


class TCNSearchSpace:
    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) \
            -> CS.ConfigurationSpace:

        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'batch_size', lower=1, upper=512, default_value=64, log=False,
            ),
            CS.UniformFloatHyperparameter(
                'dropout', lower=0.0, upper=0.5, default_value=0.0, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'kernel_size', lower=8, upper=64, default_value=8, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'num_channels', lower=16, upper=512, default_value=64, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'num_levels', lower=1, upper=5, default_value=3, log=False
            ),
        ])
        return cs
