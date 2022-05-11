from typing import List, Union, Dict, Any, Tuple
from pathlib import Path
import json
import logging

import pandas as pd
from collections import defaultdict
import numpy as np


class _BaseResultReader:
    """
    Depending on the optimizer and experiment, we have different formatted output files.
    BOHB returns a configs.json and results.json file, DEHB has history in a file called dehb_log.json.
    Our experiments by default, return a EEG_Result.json file.

    All of them return more or less the same information. This class here is responsible for bringing the
    results into the same shape.
    This is basically a pandas Dataframe storing the run results and one containing the configurations.
    These two dataframes can be merged on the configuration id (`config_id`).
    """
    def __init__(self):
        self.logger = logging.getLogger(__class__.__name__)
        super(_BaseResultReader, self).__init__()

    def load_data(self, result_files: Union[List[Union[str, Path]], str, Path]) -> Tuple:
        """
        Base method loading the data from a collection of result files.
        The results of the given result_files are collected and combined into a single runhistory.
        Make sure that they are from the same optimizer and you really want to combine them.

        Parameters
        ----------
        result_files : str, Path, List[Union[str, Path]]

        Returns
        -------

        """

        if isinstance(result_files, (str, Path)):
            result_files = [result_files]
        result_files = [Path(path) for path in result_files]

        # Load the json data from the files.
        data_files = []
        for json_file in result_files:
            data = self._load_json_results_from_file(json_file)
            data_files.append(data)

        # Collect some statistics such as the time spend or the number of target execution algorithm calls.
        stats_tae = defaultdict(lambda: 0)
        stats_costs = defaultdict(lambda: 0)  # This one contains also crashed runs.
        stats_costs_successful = defaultdict(lambda: 0)  # This one contains only successful runs.
        df = []

        for data in data_files:
            new_entries = self._format_entries(data, stats_tae, stats_costs, stats_costs_successful)
            df.extend(new_entries)

        df = pd.DataFrame(df)
        df = df.sort_values(by=['finish_time_epoch', 'start_time_epoch'])

        # Difference in run time spent for crashed runs
        stats_costs = pd.DataFrame.from_dict(stats_costs, orient='index')
        stats_costs_successful = pd.DataFrame.from_dict(stats_costs_successful, orient='index')
        diff_time = stats_costs - stats_costs_successful
        self.logger.info(f'Total time used (SUCCESSFUL):           {stats_costs.values.sum() / 3600}h')
        self.logger.info(f'Total time used (SUCCESSFUL + CRASHED): {stats_costs_successful.values.sum() / 3600}h')

        # Create a mapping from Configuration to Configuration ID.
        unique_configs = pd.DataFrame(df.loc[:, 'config'].unique(), columns=['config'])
        unique_configs.index = unique_configs.index.rename('config_id')

        # Remove the config from the table and replace it with a config_id. This is to increase readability.
        run_results = pd.merge(left=df, right=unique_configs.reset_index(), on="config")
        run_results = run_results.loc[:, run_results.columns != 'config']
        run_results = run_results.sort_values(by=['finish_time_epoch', 'start_time_epoch'], ignore_index=True)

        return run_results, unique_configs, stats_tae, stats_costs

    def _load_json_results_from_file(self, json_file: Path) -> List:
        self.logger.debug(f'Load json files from {json_file}')
        try:
            with json_file.open('r') as fh:
                data = json.load(fh)
        except json.decoder.JSONDecodeError:
            with json_file.open('r') as fh:
                lines = fh.readlines()
            data = [json.loads(line) for line in lines]
        return data

    def _format_entries(self,
                        run_data: List,
                        stats_tae: defaultdict,
                        stats_costs: defaultdict,
                        stats_costs_successful: defaultdict) -> List:

        raise NotImplementedError()


class DEHBResultReader(_BaseResultReader):
    def _format_entries(self,
                        run_data: List,
                        stats_tae: defaultdict,
                        stats_costs: defaultdict,
                        stats_costs_successful: defaultdict) -> List:

        new_entries = []

        for line in run_data:
            if isinstance(line, list) and len(line) == 3:
                line = line[2]

            config = line['info']['config']
            state = line['info']['state']

            # Extract the budget from the data. Sometimes it is called fidelity and sometimes budget.
            if 'fidelity' in line:
                budget = line['fidelity']
            elif 'budget' in line['info']:
                budget = line['info']['budget']
            elif 'budget' not in line['info'] and 'fidelity' in line['info']:
                budget = line['info']['fidelity']
            else:
                raise ValueError(f'Field Budget is not given.')

            if state == 'SUCCESS':
                description = {'config': config, 'budget': budget, 'cost': line['cost'], 'fitness': line['fitness']}
                for i_cv_fold, res in line['info']['result_per_fold'].items():
                    entry = {**{'i_cv': i_cv_fold}, **description, **res}
                    new_entries.append(entry)
                stats_costs_successful[budget] += line['cost']

            stats_tae[budget] += 1
            stats_costs[budget] += line['cost']

        return new_entries


class BOHBResultReader(_BaseResultReader):
    def _format_entries(self,
                       run_data: List,
                       stats_tae: defaultdict,
                       stats_costs: defaultdict,
                       stats_costs_successful: defaultdict) -> List:
        new_entries = []

        for line in run_data:
            exception = line[4]
            if exception is not None:
                self.logger.debug(f'Skip crashed run. Exception: {exception}')
                continue

            budget = line[1]
            cost = line[3]['info']['cost']
            config = line[3]['info']['config']
            state = line[3]['info']['state']

            if state == 'SUCCESS':
                description = {'config': config, 'budget': budget, 'cost': cost, 'fitness': line[3]['loss']}
                result_per_fold = line[3]['info']['result_per_fold']
                for i_cv_fold, res in result_per_fold.items():
                    res['start_time_epoch_'] = res['start_time_epoch']
                    res['finish_time_epoch_'] = res['finish_time_epoch']
                    res['start_time_epoch'] = line[2]['started']
                    res['finish_time_epoch'] = line[2]['finished']
                    entry = {**{'i_cv': i_cv_fold}, **description, **res}
                    new_entries.append(entry)
                stats_costs_successful[budget] += cost

            stats_tae[budget] += 1
            stats_costs[budget] += cost

        return new_entries


class EEGResultReader(_BaseResultReader):
    def __init__(self):
        super(EEGResultReader, self).__init__()

    def _format_entries(self,
                        run_data: List,
                        stats_tae: defaultdict,
                        stats_costs: defaultdict,
                        stats_costs_successful: defaultdict) -> List:

        # TODO: Use the correct BudgetManager here.
        from eeg_project.utils.help_manager import TimeBudgetManager
        new_entries = []
        for line in run_data:

            # Skip crashed runs
            exception = line['info']['exception']

            if not (exception is None or (isinstance(exception, str) and (exception != '' or exception != 'None'))):
                self.logger.debug(f'Skip crashed run. Exception: {exception}')
                continue

            # The EEG results dont include the hb budgets.
            # Try to estimate it from the cv folds and the training time.
            fidelity = json.loads(line['info']['fidelity'])
            budget = TimeBudgetManager.map_fidelity_to_hb_budget(
                cv_folds=fidelity['cv_folds'], training_time_in_s=fidelity['training_time_in_s']
            )

            cost = line['cost']
            config = line['info']['config']
            state = line['info']['state']

            if state == 'SUCCESS':
                description = {'config': config, 'budget': budget, 'cost': cost, 'fitness': line['function_value']}
                result_per_fold = line['info']['result_per_fold']

                for i_cv_fold, res in result_per_fold.items():
                    entry = {**{'i_cv': i_cv_fold}, **description, **res}
                    new_entries.append(entry)

                stats_costs_successful[budget] += cost

            stats_tae[budget] += 1
            stats_costs[budget] += cost

        return new_entries


class IntermediateResultReader(_BaseResultReader):
    # noinspection PyMethodOverriding
    def load_data(self,
                  result_files: Union[List[Union[str, Path]], str, Path],
                  mapping_files: Union[List[Union[str, Path, None]], str, Path, None] = None) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:

        if isinstance(result_files, (str, Path)):
            result_files = [result_files]

        result_files = [Path(path) for path in result_files]

        dfs = []

        # Search for all .csv files in the path:
        for result_file in result_files:
            csv_files = list(result_file.rglob('metrics.csv'))
            for csv_file in csv_files:
                _hashed_name = csv_file.parent.parent.parent.parent.name
                _df = pd.read_csv(csv_file)
                _df['name'] = _hashed_name
                dfs.append(_df)

        run_histories = pd.concat(dfs, ignore_index=True)
        run_histories = run_histories.sort_values(['name', 'finish_time_epoch', 'step'])
        if mapping_files is not None:
            if isinstance(mapping_files, (str, Path)):
                mapping_files = [mapping_files]
            mapping_files = [Path(path) for path in mapping_files]

            mapping = {}
            for file in mapping_files:
                _mapping = self._load_json_results_from_file(file)
                mapping.update(_mapping)

            mapping = pd.DataFrame.from_dict(mapping, orient='index')
            mapping = mapping.drop(columns=['path'])
        else:
            mapping = None

        return run_histories, mapping

    def _format_entries(self,
                        run_data: List,
                        stats_tae: defaultdict,
                        stats_costs: defaultdict,
                        stats_costs_successful: defaultdict) -> List:
        raise NotImplementedError()
