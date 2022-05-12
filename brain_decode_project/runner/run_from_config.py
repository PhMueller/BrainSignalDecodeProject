import argparse
import logging
from pathlib import Path
from brain_decode_project.runner.runner import Runner

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str)
    parser.add_argument('--run_config', type=str)
    parser.add_argument('--custom_seed', type=int, default=None)

    args = parser.parse_args()
    run_args = Runner.load_yaml_configs(Path(args.yaml))

    if args.custom_seed is not None:
        print(f'Change seed from {run_args[args.run_config]["seed"]} to {args.custom_seed}')
        run_args[args.run_config]['seed'] = args.custom_seed

    logger.info(f'Args: {run_args}')
    logger.info(f'Run Config: {args.run_config}')

    runner = Runner()
    runner.run(
        args=run_args[args.run_config],
        unknown={},
    )
