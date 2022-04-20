import logging
import numpy as np
from hashlib import md5
from pathlib import Path
from typing import Union, Dict, List

import pandas as pd

from brain_decode_project.modules.io import dict_to_str, write_json_with_lock

logger = logging.getLogger(__name__)


class HashTool:
    @staticmethod
    def create_hash_name(configuration: Dict, fidelity: Union[Dict, None] = None):

        str_to_hash = dict_to_str(configuration)

        if fidelity is not None:
            fidelity_str = dict_to_str(fidelity)
            str_to_hash = str_to_hash + fidelity_str

        hashed = md5(str_to_hash.encode('UTF-8')).hexdigest()
        return hashed


class ResultManager:
    def __init__(self, save_dir: Path):
        """
        This tool keeps track of the result received during the optimization procedure.
        It also create a mapping from pseudo-random (hash) of configuration to run

        Parameters
        ----------
        save_dir
        """

        self.save_dir = save_dir.expanduser().absolute()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create a mapping for the run for later analysis
        self.mapping_file = self.save_dir / 'mapping_config_results.json'

        # Store here the results for the cross validation folds
        self.result_per_cv_fold = []
        self.index = []

    def get_run_result_directory(self, configuration: Dict, fidelity: Dict) -> Path:
        """
        This function creates the path to the directory that contains the run results for a
        specific run. It is described by the configuration and the fidelity.

        Parameters
        ----------
        configuration: Dict
        fidelity: Dict

        Returns
        -------
        Path: save_directory / 'intermediate_results' / hash(configuration + fidelity)
        """
        hashed_config_fidelity = HashTool.create_hash_name(configuration, fidelity)
        intermediate_results_dir = self.save_dir / 'intermediate_results' / hashed_config_fidelity
        intermediate_results_dir = intermediate_results_dir.expanduser().absolute()
        intermediate_results_dir.mkdir(parents=True, exist_ok=True)
        return intermediate_results_dir

    def update_mapping(self, configuration: Dict, fidelity: Dict) -> None:

        hashed_config_fidelity = HashTool.create_hash_name(configuration, fidelity)
        hashed_config = HashTool.create_hash_name(configuration)

        intermediate_results_dir = self.get_run_result_directory(configuration, fidelity)

        entry = {hashed_config_fidelity: {'path': str(intermediate_results_dir),
                                          'configuration': configuration,
                                          'configuration_hash': hashed_config,
                                          'fidelity': fidelity}}

        write_json_with_lock(json_data=entry,
                             output_file=self.mapping_file,
                             lock_name='config_map',
                             lock_dir=self.save_dir,
                             overwrite=False)

    def register_intermediate_result(self, intermediate_result: List[Dict], index=None):
        # We receive two lists with equal entries.
        # The first item contains the results for the dataloader_idx_0 and the second one for the
        # dataloader_idx_2. However the entries are equal despite that the first one contains
        # `train_det_mse/dataloader_idx_0` and the second one contains `valid_mse/dataloader_idx_1`
        # We merge the two lines by creating a pandas dataframe.
        intermediate = pd.DataFrame(intermediate_result).max()
        self.result_per_cv_fold.append(intermediate)

        # Save also the index of the e.g. fold.
        if index is not None:
            self.index.append(index)

    def register_final_result(
            self, configuration: Dict,
            fidelity: Dict, costs: float,
            exception_str: Union[str, List[str], None], test_metric: bool = False,
            hb_budget: int = 0
    ) -> Dict:

        hashed_config_fidelity = HashTool.create_hash_name(configuration, fidelity)
        hashed_config = HashTool.create_hash_name(configuration)
        config_str = dict_to_str(configuration)
        fidelity_str = dict_to_str(fidelity)

        self.result_per_cv_fold = pd.DataFrame(self.result_per_cv_fold)

        # Add the id of the split as index to the df.
        # This could be the i_cv_fold value from the training.
        if len(self.index) == len(self.result_per_cv_fold):
            self.result_per_cv_fold['new_index'] = self.index
            self.result_per_cv_fold = self.result_per_cv_fold.set_index('new_index')
            self.result_per_cv_fold = self.result_per_cv_fold.sort_index()

        if exception_str is None \
                or (isinstance(exception_str, List) and len(exception_str) == 0) \
                or (isinstance(exception_str, str) and exception_str == ''):

            # Get the correct metric.
            metric = 'valid_function_value' if not test_metric else 'test_function_value'
            function_value = float(self.result_per_cv_fold.loc[:, metric].mean())
            state = "SUCCESS"
            if test_metric:
                state += " EVAL"
            exception_str = ''
        else:
            if isinstance(exception_str, str):
                exception_str = [exception_str]

            function_value = 2 ** 31  # Max value
            state = "CRASHED"

        if not np.isfinite(function_value):
            state = "CRASHED"

        result = {"function_value": function_value,
                  "cost": float(costs),
                  "info": {"result_per_fold": self.result_per_cv_fold.to_dict(orient='index'),
                           "state": state,
                           "exception": exception_str,
                           "hashed_config": hashed_config,
                           "hashed_config_fidelity": hashed_config_fidelity,
                           "config": config_str,
                           "fidelity": fidelity_str,
                           "hb_budget": hb_budget,
                           }}

        output_file = self.save_dir / ('EEG-results' + ('_test' if test_metric else '') + '.json')
        write_json_with_lock(
            json_data=[result], output_file=output_file,
            lock_name='EEG-result_logger', lock_dir=self.save_dir, overwrite=False, sort_keys=True
        )
        return result
