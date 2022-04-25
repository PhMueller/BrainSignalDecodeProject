import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from importlib import import_module

import yaml

from brain_decode_project.benchmarks.compound_benchmark import ComposedBenchmark
from brain_decode_project.benchmarks import TUH_DEBUG_SETTINGS, HGD_DEBUG_SETTINGS
from brain_decode_project.modules.io import transform_unknown_params_to_dict

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class Runner:
    @staticmethod
    def parse_args(args=None) -> Tuple[argparse.Namespace, Dict]:

        # Extract the name of all available runner objects from the library.
        runner_module = import_module('brain_decode_project.benchmarks.__init__')
        choices = [k for k in runner_module.__dict__.keys() if k.lower().endswith('Benchmark')]

        parser = argparse.ArgumentParser()
        parser.add_argument("--name", type=str, choices=choices)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--hb_budget", type=int, required=False, default=243)
        parser.add_argument("--i_cv_folds", nargs="+", type=int, required=False, default=None)
        parser.add_argument("--time_limit_in_s", type=int, default=None, help="Train each run for X seconds.")
        parser.add_argument("--epoch_limit", type=int, default=None, help="Train each run for X epochs.")
        parser.add_argument("--n_recordings_to_load", nargs="+", type=int, default=None)
        # parser.add_argument(
        #     "--config_ids", nargs="+", type=int, required=False, default=None,
        #     help="Starting from 0. Position of config in json file. Defaults to all available splits.",
        # )
        parser.add_argument("--custom_checkpoint", type=str, default=None)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--run_test', action="store_true", default=False, required=False)
        parser.add_argument('--only_healthy', action="store_true", default=False)
        parser.add_argument('--skip_training', action="store_true", default=False, required=False)
        parser.add_argument('--load_model', action="store_true", default=False, required=False)
        parser.add_argument("--debug", action="store_true", default=False, required=False)

        subparsers = parser.add_subparsers(help="Run Network", dest="entry_point")

        # Select with the next action the way to select the configuration to train
        parser_yaml = subparsers.add_parser("load_yaml")
        parser_yaml.add_argument("--yaml_file", type=str, required=True, help="yaml file.")
        parser_yaml.add_argument("--yaml_key", type=str, required=True, default='incumbent_cfg',
                                 help="key in the yaml file. Default=incumbent_cfg")

        # parser_json = subparsers.add_parser("load_json")
        # parser_json.add_argument("--json_file", type=str, required=True, help="json file.")

        parser_default = subparsers.add_parser("default_config")

        args, unknown = parser.parse_known_args(args=args)
        unknown = transform_unknown_params_to_dict(unknown)

        logger.info(f'Known:   {args}')
        logger.info(f'Unknown: {unknown}')
        return args, unknown

    @staticmethod
    def load_json_configs(json_file: Path) -> List[Dict]:
        with json_file.open("r") as fh:
            configs = json.load(fh)
        return configs

    @staticmethod
    def load_yaml_configs(yaml_file: Path) -> Dict:
        with yaml_file.open('r') as fh:
            configs = yaml.load(fh, Loader=yaml.FullLoader)
        return configs

    def run(self, args, unknown, load_model: bool = False) -> None:

        logger.info(f'Start Benchmark: {args["name"]}')

        # Import the benchmark object
        module = __import__('brain_decode_project.benchmarks', fromlist=[args["name"]])
        benchmark_obj = getattr(module, args["name"])

        output_dir = Path(args['output_dir'])
        data_dir = Path(args['data_dir'])

        if args.get('custom_checkpoint', None) is not None:
            args['custom_checkpoint'] = Path(args['custom_checkpoint'])

        load_model = load_model or args['load_model']

        # ------------------------ Load Configurations -------------------------------------------
        # Depending on the passed arguments
        #   TODO

        # if args.entry_point == "load_json":
        #     configurations = self.load_json_configs(Path(args.json_file))
        if args['entry_point'] == 'load_yaml':
            configuration = self.load_yaml_configs(Path(args['yaml_file']))
            configuration = configuration[args['yaml_key']]

        elif args['entry_point'] == "default_config":
            config_space = benchmark_obj.get_configuration_space()
            configuration = config_space.get_default_configuration().get_dictionary()
        else:
            config_space = benchmark_obj.get_configuration_space()
            configuration = config_space.get_default_configuration().get_dictionary()

        # ------------------------ Initialize the Network ----------------------------------------
        DEBUG_SETTINGS = HGD_DEBUG_SETTINGS \
            if "hgd" in self.__class__.__name__.lower() else TUH_DEBUG_SETTINGS

        # In case it is a HGD benchmark: we treat a given int as a subject id.
        # For the tuh data sets, we interpret it the number of subjects to load.
        # In both cases, we take care of it later in the data_manager. When multiple subject ids
        # are given (list), then we are sure that we mean subject ids here.
        n_recordings_to_load = args.get('n_recordings_to_load', None)
        if n_recordings_to_load is not None \
                and isinstance(n_recordings_to_load, list) and len(n_recordings_to_load) == 1:
            n_recordings_to_load = n_recordings_to_load[0]

        # When nothing is specified. -> Fall back to debug settings or load all.
        if n_recordings_to_load is None and args.get('debug', False):
            n_recordings_to_load = DEBUG_SETTINGS["n_recordings_to_load"]

        # additional = {}
        # if args.only_healthy:
        #     additional['only_healthy'] = True
        benchmark: ComposedBenchmark = benchmark_obj(
            data_path=data_dir,
            result_path=output_dir,
            # n_recordings_to_load=n_recordings_to_load,
            rng=args['seed'],
            # **additional
        )

        # ------------------------ Evaluate the Configurations -----------------------------------
        # Call the objective function of the benchmark with the given configuration on the
        # specified cv splits.
        logger.info(f"Start with configuration.")
        if not args['run_test']:
            benchmark.objective_function(
                configuration,
                fidelity={"hb_budget": args['hb_budget']},
                rng=args['seed'],
                custom_training_time_in_s=args['time_limit_in_s'],
                custom_training_epoch_limit=args['epoch_limit'],
                custom_cv_folds=args['i_cv_folds'],
                load_model=load_model,
                skip_training=False,
                disable_checkpoints=False,
                custom_checkpoint_dir=args.get('custom_checkpoint', None),
                n_recordings_to_load=n_recordings_to_load,
                debug=args.get('debug', False),
                **unknown
            )

        if args['run_test']:
            benchmark.objective_function_test(
                configuration,
                fidelity={"hb_budget": args['hb_budget']},
                rng=args['seed'],
                custom_training_time_in_s=args['time_limit_in_s'],
                custom_training_epoch_limit=args['epoch_limit'],
                custom_cv_folds=args['i_cv_folds'],
                debug=args.get('debug', False),
                load_model=load_model,
                skip_training=args['skip_training'],
                custom_checkpoint_dir=args.get('custom_checkpoint', None),
                **unknown
            )


def main():
    runner = Runner()
    args, unknown = runner.parse_args()
    runner.run(args=args, unknown=unknown, load_model=False)


if __name__ == '__main__':
    main()