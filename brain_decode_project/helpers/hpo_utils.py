"""
This file does create the BOHB run results as pkl file by converting the BOHB json files.
It also renames some configurations. Because HPBandster runs into a problem when a runs is restarted multiple times.
HPBandster gives the old results a bracket id of -1. However, if there is already a -1, this causes some errors.
As a fix, set the bracket id of the previous run results with an id of -1 to a unused value.
"""

import logging
import pickle

import numpy as np
from pathlib import Path
from hpbandster.core.result import logged_results_to_HBS_result, Result
from typing import Union

logger = logging.getLogger(__name__)


def json_to_bohb_pkl(input_dir: Path, output_file: Path, overwrite: bool = False) -> None:
    """
    Create a BOHB Result object from the bohb run results in json format.

    Parameters
    ----------
    input_dir
    output_file
    overwrite
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    assert (input_dir / 'configs.json').exists()
    assert (input_dir / 'results.json').exists()

    hpbandster_result = logged_results_to_HBS_result(str(input_dir))

    assert overwrite or (not overwrite and not output_file.exists()), \
        'The file does already exists. Either manually set the `overwrite` flag or move the ' \
        'existing file.'

    print(output_file.parent.exists())
    with open(output_file, 'wb') as fh:
        pickle.dump(hpbandster_result, fh)

    logger.info(f'Wrote Hpbandster Result to: {output_file}')


def reset_configuration_ids(previous_bohb_result: Result) -> Result:
    """
    BOHB saves every run with a corresponding bracket id. When it restarts a run, it gives
    "old" configurations a new run_id of -1. This crashes when you restart a run 2 times due to
    the presence of runs with a run_id of -1.

    Set the runs with a -1 to a conflict free configuration id.

    Parameters
    ----------
    previous_bohb_result : Result
        BOHB result object that is created during the optimization run.

    Returns
    -------
    Result
    """
    # Extract all bracket ids.
    bracket_ids = np.array([entry[0] for entry in previous_bohb_result.data.keys()])

    # If no -1 is present, this means that the opt runs has not been restarted yet.
    if -1 in bracket_ids:
        logger.info(
            'We found old runs with a -1 run_id. To avoid an error with HPBANDSTER and restarting,'
            ' set them to a different value.'
        )

        # Set the configurations with the bracket id to a not-yet-used id.
        # Here: the minimal value - 1.
        min_value = np.min(bracket_ids)
        new_dict = {}
        for key, entry in previous_bohb_result.data.items():
            v1, v2, v3 = key
            v1 = v1 if v1 != -1 else min_value - 1
            new_dict[(v1, v2, v3)] = entry
        previous_bohb_result.data = new_dict
    logger.info('Previous Result restored.')
    return previous_bohb_result


def create_new_pkl(
        new_pkl_file: Union[Path, str],
        path_to_json_files: Union[Path, str, None] = None,
        path_to_pkl_file: Union[Path, str, None] = None,
) -> None:
    """
    Helper function: Reset the configuration ids by either reading in a bohb result as pkl, or
    from the two json files.
    """
    logger.info('Load previous run results.')
    assert not (path_to_pkl_file is not None and path_to_json_files is not None)
    assert not (path_to_pkl_file is None and path_to_json_files is None)

    # if no pkl file was created, uncomment those lines, set the correct path
    # (dir in which the json files are).
    if path_to_json_files is not None:
        pkl = logged_results_to_HBS_result(path_to_json_files)
    else:
        with open(path_to_pkl_file, 'r') as fh:
            pkl = pickle.load(fh)

    # Make the Run Result object ready for restarting the run.
    reset_configuration_ids(previous_bohb_result=pkl)

    # Finalize this procedure by saving the new result file as pickle (pkl) file.
    with open(new_pkl_file, 'wb') as fh:
        pickle.dump(pkl, fh)
