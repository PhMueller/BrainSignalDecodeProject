from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter
from typing import Dict


class BaseBudgetManager:
    default_fidelity_values = {
        'num_epochs': 10000000000,
        'training_time_in_s': 10000000000,
        'n_cv_folds': 1,
        'i_cv_fold': 0,
    }

    @staticmethod
    def get_fidelity_space() -> ConfigurationSpace:
        raise NotImplementedError()

    def get_fidelity_from_hb_budget(self, fidelity: Dict) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def _get_default_fidelity():
        return BaseBudgetManager.default_fidelity_values.copy()


class DummyBudgetManager(BaseBudgetManager):
    @staticmethod
    def get_fidelity_space() -> ConfigurationSpace:
        fs = ConfigurationSpace()
        fs.add_hyperparameter(UniformIntegerHyperparameter('fidelity', lower=1, upper=12))
        return fs

    def get_fidelity_from_hb_budget(self, fidelity: Dict) -> Dict:
        f = BaseBudgetManager._get_default_fidelity()
        f.update(fidelity)
        return f
