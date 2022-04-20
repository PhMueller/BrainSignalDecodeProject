import logging
from pathlib import Path
from typing import Any, Union, Iterable, List, Text, Tuple, Dict

import numpy as np
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.preprocessing.preprocess import preprocess, Preprocessor

from brain_decode_project.data.base_data import BaseData, BaseDataSplitter, NORMALIZATION_FUNCS


import random

import ConfigSpace as CS
import torch
import torchmetrics
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, \
    exponential_moving_demean, exponential_moving_standardize
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.training import trial_preds_from_window_preds

from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset


logger = logging.getLogger(__name__)


class TUHData(BaseData):
    def __init__(
            self,
            data_path: Union[str, Path],
            n_recordings_to_load: Union[int, Iterable, None],
            data_target_name: Text,
            cut_off_first_k_seconds: int,
            n_max_minutes: int,
            sfreq: int,
            rng: int,
            train_or_eval: Union[str, None] = 'both',
            only_healthy: Union[bool, None] = False,
            standardization_op: Union[Text, None] = 'exponential_moving_demean',
    ):
        """
        Data class for the task on the TUH Abnormal Data Set.

        Parameters
        ----------
        data_path: Path
            The data has to be here. See constants.py for an example.

        n_recordings_to_load: Union[int, Iterable, None]
            This value describes how many recording are loaded.
            This value can be of type:
            - Integer: Load the first K recordings.
            - Interable: Collection of ids. Load the recordings with these ids.
            - None: Load all 2993 recordings. Default.

        data_target_name: Text
            Describes the target that is selected from the data set.
            Either age, gender, or pathological

        cut_off_first_k_seconds: int
            The first seconds of a recording tend to be noisy and contain many outliers.
            Thus, we remove the first K seconds from each recording.

        n_max_minutes: int
            Use the first `n_max_minutes` of each recording.

        sfreq: int
            Re-sample the recording to this frequency. The original data has a frequency of 250Hz.

        rng: int
            Random Seed

        train_or_eval: Text
            Use either the `train`, the `eval` or `both` splits. Defaults to `both`

        only_healthy: bool
            When this parameter is set to True, then use only subjects that do not have a reported
            illness. We observed that the models seem to classify reach better performances when
            only using only `healthy` subjects.

        standardization_op: Text
            The standardization operation that is applied to each recording.
            Has to be one of exponential_moving_demean, exponential_moving_standardize, None.
            Defaults to exponential_moving_demean. We have achieved the best performance with this
            operation.
        """

        super(TUHData, self).__init__(
            data_path=data_path,
            n_recordings_to_load=n_recordings_to_load,
            n_max_minutes=n_max_minutes,
            cut_off_first_k_seconds=cut_off_first_k_seconds,
            sfreq=sfreq,
            rng=rng
        )

        # Sometimes, we'd like to select only patients that do not have a illness.
        # This is used for a ablation experiment.
        self.only_healthy = only_healthy

        # Select the correct data target. We have three potential choices.
        _choices = ['age', 'gender', 'pathological']
        assert data_target_name in _choices, \
            'Parameter `data_target_name` unknown. ' \
            f'Has to be one of {_choices}, but was {data_target_name}'
        self.data_target_name = data_target_name

        # We apply a normalization operation to the split. This can have a significant impact on
        # the final performance. Using the `demean` has shown the best results in our experiments.
        _choices = ['exponential_moving_demean', 'exponential_moving_standardize', 'None', None]
        assert standardization_op in _choices, \
            'Parameter `standardization_op` unknown. ' \
            f'Has to be one of {_choices}, but was {standardization_op}'
        self.standardization_op = standardization_op

        # We can either load either only the train or the test split. `both` creates both splits.
        _choices = ['both', 'train', 'eval']
        assert train_or_eval in _choices, \
            'Parameter `train_or_eval` unknown. ' \
            f'Has to be one of {_choices}, but was {train_or_eval}.'

        # ------------------------- LOAD BASE DATA SET -------------------------------------------
        # We start to load the base data set by using the braindecode implementation.

        data_path = str(self.data_path)
        data_path += '/' if not data_path.endswith('/') else ''

        logger.debug(
            f'Load base data from {data_path} with n_recordings_to_load={n_recordings_to_load}'
        )

        self.base_dataset = TUHAbnormal(
            path=data_path,
            recording_ids=self.n_recordings_to_load,
            target_name=self.data_target_name,
            preload=False,
            add_physician_reports=False
        )

        # ------------------------- CREATE SPLIT -------------------------------------------------
        # Then, we extract the correct split form the base data  set and apply the preprocessing.

        logger.debug(f'Load split {train_or_eval} from base data set.')
        if train_or_eval == 'both' or train_or_eval == 'train':
            self.base_data_train, self.y_mean, self.y_std = self.load_split('train')
        if train_or_eval == 'both' or train_or_eval == 'eval':
            self.base_data_eval, _, _ = self.load_split('eval')

        logger.info('Finished loading data')

    @staticmethod
    def get_ordered_channel_names() -> List[str]:
        """
        Each channel has a unique name. These channel names correspond to the sensors used
        during the recording.

        Returns
        -------
        List
        """
        ch_names = [
            'EEG FP2-REF', 'EEG FP1-REF', 'EEG F4-REF', 'EEG F3-REF', 'EEG C4-REF', 'EEG C3-REF',
            'EEG P4-REF', 'EEG P3-REF', 'EEG O2-REF', 'EEG O1-REF', 'EEG F8-REF', 'EEG F7-REF',
            'EEG T4-REF', 'EEG T3-REF', 'EEG T6-REF', 'EEG T5-REF', 'EEG A2-REF', 'EEG A1-REF',
            'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
        ]
        ch_names = sorted(ch_names)
        return ch_names

    def load_split(self, split) -> Tuple[Any, float, float]:
        """
        Extract either the train or evaluation split from the TUH Abnormal Corpus.
        This function also applies a list of preprocessing operations.

        We preprocess both the train and the evaluation split equally with the same preprocessing
        operations.

        Parameters
        ----------
        split : string
            either "train" or "eval"

        Returns
        -------
        Dataset - the specified split of the data set.
        Mean    - the mean of the targets of the specified split
        Std     - The standard deviation of the targets of the specified split
        """

        assert split in ['train', 'eval'], 'Parameter `split` unknown. ' \
            f'Has to be one of [train, eval], but was {split}'

        # Since the braindecode version 0.5.1, we have to split along the column `train`.
        # This column has now values 'True', 'False' instead of 'train', 'eval'.
        split = 'True' if split == 'train' else 'False'

        # Extract the correct split from the base data set.
        whole_set = self.base_dataset.split(by='train')[split]

        if self.only_healthy:
            whole_set = whole_set.split(by='pathological')['False']

        preprocessors = self.get_preprocessing_pipeline()
        preprocess(whole_set, preprocessors)

        # Extract the target and compute the mean and std. We use these values to normalize
        # the age targets.
        targets = np.array([ds.description[ds.target_name] for ds in whole_set.datasets])

        if self.data_target_name == 'age':
            y_mean, y_std = targets.mean(), targets.std()
        else:
            y_mean, y_std = 0, 1

        # In case it is the 'GENDER' data set, replace 'F' with 1 and 'M' with 0
        if self.data_target_name == 'gender':
            for ds in whole_set.datasets:
                ds.description[ds.target_name] = 1 if ds.description[ds.target_name] == 'F' else 0

        # In case it is the 'PATHOLOGICAL' data set, we encode the presence of a illness
        # with 1, 0 otherwise.
        if self.data_target_name == 'pathological':
            for ds in whole_set.datasets:
                ds.description[ds.target_name] = 1 if ds.description[ds.target_name] else 0

        logger.info(f'Dataset: {split}. Y_mean = {y_mean}. Y_std = {y_std}')
        return whole_set, y_mean, y_std

    def get_preprocessing_pipeline(self) -> List:
        """
        We define a set of preprocessing operations that are applied to each split. 

        Returns
        -------
        List with preprocessing operations
        """
        preprocessors = [
            Preprocessor(
                fn='pick_channels', ch_names=self.ch_names, ordered=True, apply_on_array=False
            ),

            # Following
            # https://www.isip.piconepress.com/publications/ms_theses/2017/abnormal/thesis/
            # Physicians can distinguish normal from abnormal from the first few seconds.
            # Choose here the first X minutes. (A recording is ~20 min in total long)
            Preprocessor(
                'crop',
                tmin=self.cut_off_first_k_seconds,
                tmax=self.n_max_minutes * 60,
                include_tmax=True,
                apply_on_array=False
            ),

            # Convert from volt to microvolt
            Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),

            # Clip the signal between -800 and 800 microVolt
            Preprocessor(fn=lambda x: np.clip(x, -800, 800), apply_on_array=True),

            # This is defined by Schirrmeister et al.
            Preprocessor(fn=lambda x: x / 30, apply_on_array=True),

            # Resample the signal to have a frequency of `sfreq`
            Preprocessor(fn='resample', sfreq=self.sfreq, apply_on_array=False),
        ]

        # We standardize the split using a demean or a 0-mean-unit-variance standardization.
        # But we also allow to have no standardization at all.
        if not self.standardization_op in ['None', None]:

            preprocessors.append(
                Preprocessor(
                    fn=NORMALIZATION_FUNCS[self.standardization_op],
                    # Schirrmeister et al found these values empirically.
                    init_block_size=int(self.sfreq * 10), factor_new=1 / (self.sfreq * 5),
                    apply_on_array=True
                )
            )

        return preprocessors


class TUHDataSplitter(BaseDataSplitter):
    def __init__(
            self,
            input_window_samples,
            window_stride_samples
    ):
        super(TUHDataSplitter, self).__init__(
            input_window_samples=input_window_samples,
            window_stride_samples=window_stride_samples,
        )

    def split_into_train_valid(
        self,
        dataset,
    ) -> Tuple[WindowsDataset, WindowsDataset]:
        """
        Create the training and validation split.

        This functions splits the TUH Training split into two subsets.
        80% train and 20% valid.

        Parameters
        ----------
        dataset: TUHData

        Returns
        -------
        Tuple[WindowsDataset, WindowsDataset]
        """
        logger.info('Use the Train - Validation split')

        # Split the train dataset into Train and Valid
        subject_datasets = dataset.base_data_train.split('subject')
        n_subjects = len(subject_datasets)
        n_split = int(np.round(n_subjects * 0.8))
        keys = list(subject_datasets.keys())
        random.shuffle(keys)

        train_sets = [d for i in range(n_split) for d in subject_datasets[keys[i]].datasets]
        train_set = BaseConcatDataset(train_sets)
        valid_sets = [d for i in range(n_split, n_subjects) for d in
                      subject_datasets[keys[i]].datasets]
        valid_set = BaseConcatDataset(valid_sets)

        train_windows_dataset = create_fixed_length_windows(
            concat_ds=train_set,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=self.input_window_samples,
            window_stride_samples=self.window_stride_samples,
            drop_last_window=False,
            preload=True,
        )

        valid_windows_dataset = create_fixed_length_windows(
            concat_ds=valid_set,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=self.input_window_samples,
            window_stride_samples=self.window_stride_samples,
            drop_last_window=False,
            preload=True,
        )

        return train_windows_dataset, valid_windows_dataset

    def split_into_final_train_valid(
            self,
            dataset,
    ) -> Tuple[WindowsDataset, WindowsDataset]:
        """
        Create the training and test split.

        The training split contains all available recordings from the training subset.
        The test split is a holdout set.

        Parameters
        ----------
        dataset: TUHData

        Returns
        -------
        Tuple[WindowsDataset, WindowsDataset]
        """
        logger.info('Use the Train - Test split')

        # -------------------- Train Split -------------------------------------------------------
        # Use the entire train data split. Shuffle the subjects according to the given seed
        train_subject_datasets = dataset.base_data_train.split('subject')

        n_subjects = len(train_subject_datasets)
        keys = list(train_subject_datasets.keys())
        random.shuffle(keys)

        train_sets = [d for i in range(n_subjects) for d in
                      train_subject_datasets[keys[i]].datasets]
        train_set = BaseConcatDataset(train_sets)

        # -------------------- Test Split --------------------------------------------------------
        # Here we use the test split. This is a predefined holdout test split
        test_subject_datasets = dataset.base_data_eval.split('subject')
        n_subjects = len(test_subject_datasets)
        keys = list(test_subject_datasets.keys())
        random.shuffle(keys)

        test_sets = [d for i in range(n_subjects) for d in
                     test_subject_datasets[keys[i]].datasets]
        test_set = BaseConcatDataset(test_sets)

        train_windows_dataset = create_fixed_length_windows(
            concat_ds=train_set,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=self.input_window_samples,
            window_stride_samples=self.window_stride_samples,
            drop_last_window=False,
            preload=True,
        )

        test_windows_dataset = create_fixed_length_windows(
            concat_ds=test_set,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=self.input_window_samples,
            window_stride_samples=self.window_stride_samples,
            drop_last_window=False,
            preload=True,
        )
        return train_windows_dataset, test_windows_dataset