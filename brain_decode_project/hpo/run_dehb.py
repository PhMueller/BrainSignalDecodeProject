import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import List

from dehb import DEHB
from distributed import Client

from brain_decode_project.helpers.benchmark_utils import is_benchmark_available, \
    get_available_benchmarks, get_benchmark_object
from brain_decode_project.helpers.env_utils import remove_slurm_env_variables
from brain_decode_project.hpo.worker import dehb_objective_function

DEBUG = False

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

main_logger = logging.getLogger('DEHB-Main')


def run_optimization(args):

    # We need a benchmark name. It has to be one of the defined benchmarks
    assert args.benchmark_name is not None and is_benchmark_available(args.benchmark_name), \
        f'Parsed Benchmark was {args.benchmark_name}, but has to be one of ' \
        f'{[get_available_benchmarks()]}'

    benchmark_obj = get_benchmark_object(benchmark_name=args.benchmark_name)
    complete_cs = benchmark_obj.get_configuration_space(args.seed)

    # In the local case, we don't need a client and only start a single worker.
    if args.local_run:
        main_logger.info('This run is started as a local run. Skip looking for scheduler files.')
        n_workers = 1
        client = None

    # In the distributed case, we let the system determine its number of workers on its own.
    else:
        # The scheduler file has to exist
        if args.scheduler_file is None or not Path(args.scheduler_file).is_file():
            scheduler_file = args.scheduler_file or ''
            main_logger.warning(
                f'Scheduler File: {str(scheduler_file)}. '
                f'File exists? {Path(scheduler_file).is_file()}'
            )
            raise ValueError('You have to specify a scheduler file. But was not found.')
        n_workers = None
        client = Client(scheduler_file=args.scheduler_file)
        if args.min_n_workers > 0:
            main_logger.info(
                f'Wait for at least {args.min_n_workers} before starting the optimization run.'
            )
            client.wait_for_workers(n_workers=args.min_n_workers)

    result_path = Path(args.result_path)
    data_path = Path(args.data_path)
    checkpoint_file = result_path / 'checkpoint.pkl'
    seed = args.seed

    main_logger.info(f'Start Dehb with the args {args}')

    dehb = DEHB(
        f=dehb_objective_function,
        cs=complete_cs,
        dimensions=len(complete_cs.get_hyperparameters()),
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta,
        output_path=result_path / 'dehb_logs',
        client=client,
        client_resources={'limit_proc': 1},
        restore_checkpoint=args.restore_checkpoint,
        checkpoint_file=checkpoint_file,
        n_workers=n_workers
    )

    try:
        traj, runtime, history = dehb.run(
            brackets=args.brackets,
            total_cost=args.runtime,
            verbose=args.verbose,
            save_intermediate=True,
            # arguments below are part of **kwargs shared across workers
            eta=args.eta,
            result_path=result_path,
            data_path=data_path,
            rng=seed,
            benchmark_name=args.benchmark_name,
            n_recordings_to_load=args.n_recordings_to_load
        )

    except Exception:
        dehb.save_checkpoint(
            checkpoint_file=checkpoint_file
        )
        dehb.save_checkpoint(
            checkpoint_file=checkpoint_file.parent / 'checkpoint_after_crash.pkl'
        )
        raise ValueError(
            'Crash has occured! Please resolve the error and restart the optimization run!'
        )

    dehb.save_checkpoint(checkpoint_file)
    save_dehb_history(result_path, dehb.start, [traj, runtime, history])


def save_dehb_history(result_path: Path, start_time: float, history: List) -> None:
    main_logger = logging.getLogger('DEHB-Main')

    name = time.strftime("%x %X %Z", time.localtime(start_time))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')

    main_logger.info("Saving optimisation trace history...")
    result_file = result_path / f'history_{name}.pkl'
    with result_file.open("wb") as f:
        pickle.dump(history, f)


def parser_dehb_arguments() -> argparse.Namespace:
    """
    Parse the commandline arguments for all dehb experiments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Optimizing a model using DEHB.')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--refit_training', action='store_true', default=False,
                        help='Refit with incumbent configuration on full training data and '
                             'budget')
    parser.add_argument('--min_budget', type=float, default=None,
                        help='Minimum budget (epoch length)')
    parser.add_argument('--max_budget', type=float, default=None,
                        help='Maximum budget (epoch length)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Parameter for Hyperband controlling early stopping aggressiveness')
    parser.add_argument('--scheduler_file', type=str, default=None,
                        help='The file to connect a Dask client with a Dask scheduler')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Folder that contains the data')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Folder in that we store the run results')
    parser.add_argument('--restore_checkpoint', action='store_true', default=False)
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Decides verbosity of DEHB optimization')
    parser.add_argument('--runtime', type=float, default=None, required=False,
                        help='Total time in seconds as budget to run DEHB')
    parser.add_argument('--brackets', type=int, default=None,
                        help='Total time in seconds as budget to run DEHB')
    parser.add_argument('--local_run', action='store_true', default=False, required=False,
                        help='Whether to start the Opt run wit or without worker (locally).')
    parser.add_argument('--benchmark_name', default=None, required=False,
                        help='Benchamrk name e.g. AgeTCNBenchmark. '
                             'See the BenchmarkEnum for all options.')
    parser.add_argument('--min_n_workers', default=0, type=int,
                        help='Wait for at least N workers before starting the optimization '
                             'procedure. If set to 0 (default value), start without waiting.')
    parser.add_argument('--n_recordings_to_load', type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    remove_slurm_env_variables()
    args = parser_dehb_arguments()

    run_optimization(args)
