import json
import logging

from pathlib import Path
from typing import Union, Dict, List, Any, Optional

import pandas as pd
from oslo_concurrency import lockutils


logger = logging.getLogger(__name__)


def transform_unknown_params_to_dict(unknown_args: List) -> Dict:
    """
    Given a list of unknown parameters in form ['--name', '--value', ...], it transforms the
    list into a dictionary of shape {'name': 'value', ... }

    Parameters
    ----------
    unknown_args : List

    Returns
    -------
    Dict
    """
    unknown_params = {}
    for i in range(0, len(unknown_args), 2):
        try:
            value = int(unknown_args[i+1])

        # We can't convert it to an integer. In this case we treat it as it is (str).
        except ValueError:
            value = unknown_args[i+1]

        except IndexError:
            raise IndexError(
                'While parsing additional arguments an index error occurred. This means a '
                'parameter has no value.'
            )

        unknown_params[unknown_args[i][2:]] = value
    return unknown_params


def save_dataframe(df: pd.DataFrame, output_file: Path, key: str = 'epochs_df'):
    """ Save a pandas dataframe to file in hdf format. """
    df.to_hdf(str(output_file), key, mode='w')


def read_dataframe(file: Union[str, Path], key: str = 'epochs_df'):
    """ Save a pandas dataframe from file in hdf format. """
    return pd.read_hdf(file, key)


def dict_to_str(dictionary: Dict):
    dictionary = json.dumps(dictionary, sort_keys=True)
    return dictionary


def write_json_with_lock(json_data: Union[Dict, List], output_file: Path, lock_name: str, lock_dir: Path,
                         overwrite: Optional[bool] = True, sort_keys: Optional[bool] = True) -> None:
    """
    Append a `json_data` payload to a file (if it already exists).
    Secure this operation with a lock.

    Parameters
    ----------
    json_data : Union[Dict, List]
    output_file : Path
    lock_name: str
    lock_dir : Path
    overwrite : bool = True
        If the file already exists, append the payload to the file if overwrite is False.
        If overwrite is True, overwrite the contents of the file.
    sort_keys: bool = True
        Sort the dict keys using the json dump function. Note that this may take more time.

    Returns
    -------
    None
    """
    lock = lockutils.lock(
        name=f'{lock_name}.lock', external=True, do_log=False, lock_path=str(lock_dir)
    )
    with lock:
        if not overwrite and output_file.exists():
            logger.debug('File exists already. Going to append the data.')
            with output_file.open('r') as fh:
                old_data = json.load(fh)

            if isinstance(json_data, Dict):
                json_data = {**json_data, **old_data}
            elif isinstance(json_data, List):
                json_data.extend(old_data)

        with output_file.open('w') as fh:
            json.dump(json_data, fh, indent=4, sort_keys=sort_keys)

    """
    config_map_lock = lockutils.lock(name='config_map.lock', external=True, do_log=False, lock_path=str(result_path))
    with config_map_lock:
        with mapping_file.open('r+') as fh:
            try:
                mapping = json.load(fh)
            except json.decoder.JSONDecodeError:
                main_logger.debug('Empty mapping file. Going to create a first entry')
                mapping = dict()

            mapping.update({str(hashed_result_path): {'configuration': configuration, 'fidelity': fidelity}})
            fh.seek(0)
            json.dump(mapping, fh)
    """


def read_json_with_lock(json_file: Path, lock_name: str, lock_dir: Path) -> Union[List, Dict]:
    lock = lockutils.lock(
        name=f'{lock_name}.lock', external=True, do_log=False, lock_path=str(lock_dir)
    )
    with lock:
        with json_file.open('r') as fh:
            data = json.load(fh)
    return data
