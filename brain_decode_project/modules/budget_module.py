from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter


class BaseBudgetManager:
    pass

    @staticmethod
    def get_fidelity_space() -> ConfigurationSpace:
        raise NotImplementedError()


class DummyBudgetManager(BaseBudgetManager):
    @staticmethod
    def get_fidelity_space() -> ConfigurationSpace:
        fs = ConfigurationSpace()
        fs.add_hyperparameter(UniformIntegerHyperparameter('fidelity', lower=1, upper=12))
        return fs
