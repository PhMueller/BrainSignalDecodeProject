import logging

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter
from typing import Dict, Union, Tuple


class BaseBudgetManager:
    default_fidelity_values = {
        'num_epochs': 10000000000,
        'training_time_in_s': 10000000000,
        'n_cv_folds': 1,
        'i_cv_fold': 0,
        'hb_budget': 1,
    }

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigurationSpace:
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_from_hb_budget(fidelity: Dict) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def _get_default_fidelity():
        return BaseBudgetManager.default_fidelity_values.copy()


class DummyBudgetManager(BaseBudgetManager):
    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigurationSpace:
        fs = ConfigurationSpace(seed=seed)
        fs.add_hyperparameter(UniformIntegerHyperparameter('hb_budget', lower=1, upper=12))
        return fs

    @staticmethod
    def get_fidelity_from_hb_budget(fidelity: Dict) -> Dict:
        f = BaseBudgetManager._get_default_fidelity()
        f.update(fidelity)
        return f


class BaselineTimeBudgetManager(BaseBudgetManager):

    eta = 3
    mapping_hb_to_time_cv_folds = {
        1: (5, 1),
        3: (10, 1),
        9: (20, 1),
        27: (40, 1),
        81: (60, 3),
        243: (60, 9),
    }

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigurationSpace:
        fs = ConfigurationSpace(seed=seed)
        fs.add_hyperparameter(UniformIntegerHyperparameter('hb_budget', lower=1, upper=243))
        return fs

    @staticmethod
    def get_fidelity_from_hb_budget(fidelity: Dict) -> Dict:
        """map the hb budgets (depending on eta) [1,3,9,27,81,243] to experimental budgets """

        hb_budget = fidelity.get('hb_budget', None)
        new_fidelity = BaseBudgetManager._get_default_fidelity()
        new_fidelity.update(fidelity)

        if hb_budget is not None:
            train_time, n_cv_folds = BaselineTimeBudgetManager.mapping_hb_to_time_cv_folds[hb_budget]
            new_fidelity['training_time_in_s'] = train_time * 60
            new_fidelity['n_cv_folds'] = n_cv_folds
        else:
            logging.warning(f'HB Budget was not passed. Use Default Values. {new_fidelity}')
        return new_fidelity

    @staticmethod
    def map_fidelity_to_hb_budget(training_time_in_s: Union[int, float], cv_folds: int) -> int:
        """
        This function TRIES to map from the training time to the hb budget.
        It is only used in the analysis scripts. DO NOT use this one.
        When reading the EEG_results.json, we dont have a hint which hb_budget was used, except the training time.

        This function "fails" if a custom training time was used. In this case, we return a integer that
        is between the official hb budgets.

        E.g. if 9 folds have been used and the training time was > 3600, we return int(243 * (training time / 3600)).
        This helps to create different hb budgets. I am not sure If i will even use this.

        Parameters
        ----------
        training_time_in_s: int
        cv_folds: int

        Returns
        -------
        hb_budget - int
        """
        inverse_mapping = {v: k for k, v in BaselineTimeBudgetManager.mapping_hb_to_time_cv_folds}

        training_time_in_min = int(training_time_in_s / 60)

        hb_budget = inverse_mapping.get((training_time_in_min, cv_folds), None)

        # Given that we dont match an official budget, we need to create it from the cv and the time
        if hb_budget is None:
            hb_budget = 243 * (training_time_in_min / 60)

        return int(hb_budget)


class EpochBudgetManager(BaseBudgetManager):

    eta = 3
    mapping_hb_to_epoch_cv_folds = {
        1:   (60,   1),
        3:   (60,   1),
        9:   (180,  1),
        27:  (540,  1),
    }

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigurationSpace:
        fs = ConfigurationSpace(seed=seed)
        fs.add_hyperparameter(UniformIntegerHyperparameter('hb_budget', lower=1, upper=27))
        return fs

    @staticmethod
    def get_fidelity_from_hb_budget(fidelity: Dict) -> Dict:
        """map the hb budgets (depending on eta) [1,3,9,27,81,243] to experimental budgets """

        hb_budget = fidelity.get('hb_budget', None)
        new_fidelity = BaseBudgetManager._get_default_fidelity()
        new_fidelity.update(fidelity)

        if hb_budget is not None:
            n_train_epochs, n_cv_folds = EpochBudgetManager.mapping_hb_to_epoch_cv_folds[hb_budget]
            new_fidelity['num_epochs'] = n_train_epochs
            new_fidelity['n_cv_folds'] = n_cv_folds
        else:
            logging.warning(f'HB Budget was not passed. Use Default Values. {new_fidelity}')
        return new_fidelity

    @staticmethod
    def map_fidelity_to_hb_budget(num_epochs: int, **kwargs) -> int:
        """map the hb budgets (depending on eta) [1,3,9,27,81,243] to experimental budgets """
        inverse_mapping = {v: k for k, v in EpochBudgetManager.mapping_hb_to_epoch_cv_folds}
        num_epochs = int(num_epochs)
        hb_budget = inverse_mapping.get((num_epochs, 1), None)

        # If the hb_budget is not given, represent the hb_value as multiple of the highest budget
        if hb_budget is None:
            hb_budget = 27 * num_epochs

        return int(hb_budget)


class SingleFidelityEpochBudgetManager(BaseBudgetManager):

    # TODO: Create a 80 / 20 Split

    eta = 3
    mapping_hb_to_epoch_cv_folds = {
        1: (300, 1),
    }

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> ConfigurationSpace:
        fs = ConfigurationSpace(seed=seed)
        fs.add_hyperparameter(UniformIntegerHyperparameter('hb_budget', lower=1, upper=1))
        return fs

    @staticmethod
    def get_fidelity_from_hb_budget(fidelity: Dict) -> Dict:
        """map the hb budgets (depending on eta) [1,3,9,27,81,243] to experimental budgets """

        hb_budget = fidelity.get('hb_budget', None)
        new_fidelity = BaseBudgetManager._get_default_fidelity()
        new_fidelity.update(fidelity)

        if hb_budget is not None:
            n_train_epochs, n_cv_folds = EpochBudgetManager.mapping_hb_to_epoch_cv_folds[hb_budget]
            new_fidelity['num_epochs'] = n_train_epochs
        else:
            logging.warning(f'HB Budget was not passed. Use Default Values. {new_fidelity}')
        return new_fidelity

    @staticmethod
    def map_fidelity_to_hb_budget(num_epochs: int, **kwargs) -> int:
        if num_epochs == 300:
            hb_budget = 1
        else:
            hb_budget = int(num_epochs / 300)

        return hb_budget

    # @staticmethod
    # def create_fidelity_object(hb_budget, i_cv_fold) -> Dict:
    #     cv_folds, num_epochs, _ = \
    #         EpochBudgetManager.map_hb_fidelity_to_fidelity(hb_budget=hb_budget)
    #
    #     fidelity = {
    #         'training_epochs': num_epochs,
    #         'training_time_in_s': 1000000000 * 3600,  # Set to inifite time
    #         'n_recordings_to_load': 2993,
    #         'cv_folds': 5,  #  Create a 80 / 20 split.
    #         'i_cv_fold': i_cv_fold,
    #         'split_fraction': None
    #     }
    #
    #     return fidelity
