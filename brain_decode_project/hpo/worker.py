import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Union
import sys

from hpbandster.core.worker import Worker as BOHB_Worker

from brain_decode_project.helpers.benchmark_utils import is_benchmark_available
from brain_decode_project.helpers.logging_utils import setup_logger_for_objective_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TUHWorker(BOHB_Worker):
    def __init__(
            self, result_path, data_path, rng, benchmark_name, n_recordings_to_load=None,
            *args, **kwargs
    ):

        super(TUHWorker, self).__init__(*args, **kwargs)

        assert is_benchmark_available(benchmark_name), f'benchmark_name: {benchmark_name} unknown'

        self.rng = rng
        self.result_path = result_path
        self.data_path = data_path
        self.benchmark_name = benchmark_name
        self.n_recordings_to_load = n_recordings_to_load

    def compute(self, config: Dict, budget: Any, **kwargs) -> Dict:

        logger.debug(f'Worker is called with budget {budget}')

        data_path = str(self.data_path) if isinstance(self.data_path, Path) else self.data_path

        result_dict = _general_query(
            config, budget, str(self.result_path), str(data_path), self.rng, self.benchmark_name,
            self.n_recordings_to_load
        )

        # Cast the results to HpBandSter specific format.
        bohb_result_dict = {
            'loss': result_dict['function_value'],
            'info': {**result_dict['info'], 'cost': result_dict['cost']}
        }

        logger.debug(f'Return result with loss {result_dict["function_value"]}')

        return bohb_result_dict


def dehb_objective_function(config, budget, **kwargs):
    """

    Parameters
    ----------
    config : CS.Configuration
    budget : fidelity
    kwargs:
        Those parameters are shared across the workers. We can instantiate them in the
        main process. It could be possible to init also the data set in the main process and send
        it to the workers via this argument. However, this does not seem to be clever, since the
        data is pretty large and may cause instabilities.

    Returns
    -------
    Dict
    """

    # ------------------------------ Setup -------------------------------------------------------
    setup_logger_for_objective_function(debug=kwargs.get('debug', False))
    main_logger = logging.getLogger('DEHB-Main')
    main_logger.info(f'Called Objective: {budget} with configuration {config}')

    data_path = kwargs.get('data_path', None)
    data_path = str(data_path) if isinstance(data_path, Path) else data_path
    result_path = str(kwargs.get('result_path'))
    rng = kwargs.get('rng')

    benchmark_name = kwargs.get('benchmark_name')
    assert is_benchmark_available(benchmark_name), f'benchmark_name: {benchmark_name} unknown'

    n_recordings_to_load = kwargs.get('n_recordings_to_load', None)

    # ------------------------------ Start Optimization ------------------------------------------
    result_dict = _general_query(
        config, budget, result_path, data_path, rng, benchmark_name, n_recordings_to_load
    )

    # DEHB needs a dict that contains the field fitness value.
    result_dict['fitness'] = result_dict['function_value']

    return result_dict


def _general_query(
        config: Dict, budget: int, result_path: str, data_path: str, rng: int,
        benchmark_name: str, n_recordings_to_load: Union[int, None]
    ):
    pool = multiprocessing.Pool(processes=1)
    result_dict = pool.map(
        _subprocess_run,
        ((config, budget, result_path, data_path, rng, benchmark_name, n_recordings_to_load),)
    )  # type: List[Dict]
    pool.terminate()

    # Pool returns a list. But since we are using only a single process,
    # we always want to take the first element.
    result_dict = result_dict[0]  # type: Dict

    # Check for a misconfiguration exception:
    # (Happens on the uni freiburg cluster currently)
    for line in result_dict['info']['exception']:
        if 'MisconfigurationException' in line:
            logger.warning(
                'This Node has raised a Misconfiguration Exception. '
                'This means this Node is not useable. Stop this worker.'
            )
            sys.exit(111)
    return result_dict


def _subprocess_run(input_arguments):
    """
    This is the function that is started in a subprocess.
    It creates a benchmark object and call its objective function
    Parameters
    ----------
    input_arguments : tuple
        Consists of the configuration, fidelity, path to the result dir, path to the data, seed

    Returns
    -------
    Dict
    """

    from brain_decode_project.helpers.env_utils import set_required_env_variables

    config, budget, result_path, data_path, rng, benchmark_name, n_recordings_to_load = \
        input_arguments

    set_required_env_variables()

    result_path = Path(result_path) if isinstance(result_path, str) else result_path
    data_path = Path(data_path) if isinstance(data_path, str) else data_path

    module = __import__('brain_decode_project.benchmarks', fromlist=[benchmark_name])
    benchmark_obj = getattr(module, benchmark_name)

    benchmark = benchmark_obj(
        result_path=result_path, data_path=data_path, rng=0,
    )

    result_dict = benchmark.objective_function(
        configuration=config, fidelity=dict(hb_budget=int(budget)), rng=rng, debug=False,
        n_recordings_to_load=n_recordings_to_load,
    )

    return result_dict
