import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from shutil import move, copy
from time import sleep

from ConfigSpace.read_and_write import json as cs_json
from hpbandster.core import result as hpres, nameserver as hpns
from hpbandster.optimizers import BOHB

from brain_decode_project.helpers.benchmark_utils import is_benchmark_available, \
    get_available_benchmarks, get_benchmark_object
from brain_decode_project.helpers.env_utils import remove_slurm_env_variables
from brain_decode_project.hpo.worker import TUHWorker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

hpbandster_logger = logging.getLogger('hpbandster')
hpbandster_logger.setLevel(logging.INFO)


def run_optimization(
        benchmark_name: str,
        run_id: str,
        seed: int,
        worker_id: int,
        data_path: str,
        result_path: str,
        num_iterations: int,
        min_budget: int, max_budget: int,
        eta: int = 3,
        nic_name: str = 'en0',
        previous_results_pkl=None,
        n_recordings_to_load=None
):

    # We need a benchmark name. It has to be one of the defined benchmarks
    assert benchmark_name is not None and is_benchmark_available(benchmark_name), \
        f'Parsed Benchmark was {benchmark_name}, but has to be one of ' \
        f'{[get_available_benchmarks()]}'

    if data_path is not None:
        data_path = Path(data_path)
    result_path = Path(result_path)
    result_path.mkdir(exist_ok=True, parents=True)

    logger.info(f'Start the optimization script with worker_id: {worker_id}')

    # Worker: This runs not on the main node.
    if worker_id != 0:
        # Give the main node / nameserver some time to start
        sleep(20)

        # Start a worker. To enable the main node to communicate also with this worker,
        # look up the current ip address.
        ip_worker = hpns.nic_name_to_host(nic_name)
        worker = TUHWorker(
            run_id=run_id,
            n_recordings_to_load=n_recordings_to_load,
            rng=worker_id,
            result_path=result_path,
            data_path=data_path,
            host=ip_worker,
            benchmark_name=benchmark_name,
        )

        # Load the ip and port of the nameserver from the credential file.
        worker.load_nameserver_credentials(str(result_path))
        worker.run(background=False)
        logger.info(f'Stop worker {worker_id}')
        return

    # Main Node.
    # The result logger takes care of writing the results to the correct path.
    result_logger = hpres.json_result_logger(directory=str(result_path), overwrite=True)

    # We load the correct configuration space from the benchmark object
    # and pass it later to the optimizer.
    benchmark_obj = get_benchmark_object(benchmark_name)
    cs = benchmark_obj.get_configuration_space(seed=seed)

    # Since we need the configuration space for later analysis with CAVE, store it as json object.
    with open(result_path / 'configspace.json', 'w') as fh:
        fh.write(cs_json.write(cs))

    # Init the nameserver and start it. The nameserver is like a phonebook. Every object
    # (workers, optimizer) writes its address into it so that the objects can communicate with
    # each other.
    ns = hpns.NameServer(
        run_id=str(run_id),
        nic_name=nic_name,
        working_directory=str(result_path)
    )
    ns_host, ns_port = ns.start()

    # Start also a worker in the background, since running the main node is quite cheap.
    worker = TUHWorker(
        benchmark_name=benchmark_name,
        n_recordings_to_load=n_recordings_to_load,
        run_id=run_id,
        rng=worker_id,
        result_path=result_path,
        data_path=data_path,
        nameserver=ns_host,
        nameserver_port=ns_port,
    )
    worker.run(background=True)

    # BOHB also supports warmstarting. However, you can only warmstart a single time without some
    # renaming in the config.json / result.json file. You will know what to do when you
    # encounter the error. :-)
    previous_result = None
    if previous_results_pkl is not None:
        if not Path(previous_results_pkl).exists():
            logger.warning(f'BOHB checkpoint not found. {previous_results_pkl}. Skip restarting')
        else:
            with open(previous_results_pkl, 'rb') as fh:
                previous_result = pickle.load(fh)

            from brain_decode_project.helpers.hpo_utils import reset_configuration_ids
            previous_result = reset_configuration_ids(previous_bohb_result=previous_result)
            logger.info('Previous Result restored.')

    optimizer = BOHB(
        configspace=cs,
        run_id=run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        eta=eta,
        min_budget=min_budget,
        max_budget=max_budget,
        result_logger=result_logger,
        previous_result=previous_result
    )

    result = optimizer.run(n_iterations=num_iterations)

    logger.info('Optimization has finished. Write results to file.')

    # I recommend to save the run object (contains all the results) with pickle.
    # There are situations when it is quite useful. But in case there is already a result object,
    # rename it.
    today = datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")
    if (result_path / 'results.pkl').exists():
        move(result_path / 'results.pkl', result_path / f'results_backup_{today}.pkl')
    if (result_path / 'results.json').exists():
        copy(result_path / 'results.json', result_path / f'results_backup_{today}.json')
    if (result_path / 'configs.json').exists():
        copy(result_path / 'configs.json', result_path / f'configs_backup_{today}.json')

    with open(result_path / 'results.pkl', 'wb') as fh:
        pickle.dump(result, fh)

    optimizer.shutdown(shutdown_workers=True)
    ns.shutdown()


def parse_hpo_args():
    parser = argparse.ArgumentParser('BOHB')
    parser.add_argument('--run_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--network_interface', type=str, default='eno1')
    parser.add_argument('--worker_id', type=int)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--min_budget', type=int, default=1)
    parser.add_argument('--max_budget', type=int, default=243)
    parser.add_argument('--eta', type=int, default=3)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--data_path', type=str, default=None, required=False)
    parser.add_argument('--previous_results_pkl', type=str, required=False, default=None)
    parser.add_argument('--benchmark_name', default=None, required=False,
                        help='Benchamrk name e.g. AgeTCNBenchmark. See the BenchmarkEnum for all '
                             'options.')
    parser.add_argument('--n_recordings_to_load', type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    remove_slurm_env_variables()

    args = parse_hpo_args()

    run_optimization(
        benchmark_name=args.benchmark_name,
        run_id=args.run_id,
        seed=args.seed,
        worker_id=args.worker_id,
        nic_name=args.network_interface,
        num_iterations=args.num_iterations,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta,
        result_path=args.result_path,
        data_path=args.data_path,
        previous_results_pkl=args.previous_results_pkl,
        n_recordings_to_load=args.n_recordings_to_load
    )
