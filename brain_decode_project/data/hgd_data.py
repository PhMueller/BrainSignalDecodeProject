import logging
import os
from pathlib import Path
from typing import Union, Iterable, List, Tuple, Any, Text

import numpy as np
from braindecode.datasets.moabb import HGD
from braindecode.preprocessing.preprocess import Preprocessor, preprocess

from brain_decode_project.data.base_data import BaseData, NORMALIZATION_FUNCS

logger = logging.getLogger(__name__)


class HighGammaData(BaseData):
    def __init__(
            self,
            data_path: Union[str, Path],
            n_recordings_to_load: Union[int, Iterable, None],
            cut_off_first_k_seconds: Union[float, int],
            n_max_minutes: Union[int, None],
            sfreq: int,
            rng: int,
            do_common_average_reference: bool = False,
            low_cut_hz: Union[float, int, None] = None,
            high_cut_hz: Union[float, int, None] = None,
            standardization_op: Union[Text, None] = 'exponential_moving_standardize',
    ):
        """
        Data class for the classification task on the High Gamma Data Set.

        References:
        -----------
        [1] https://github.com/robintibor/high-gamma-dataset

        Parameters
        ----------
        data_path: Path
            The data has to be here. See constants.py for an example.

        n_recordings_to_load: Union[int, Iterable, None]
            This value describes how many recording are loaded.
            This value can be of type:
            - Integer: Interpret this value as the subject id.
            - Interable: Collection of ids. Load the recordings with these ids.
            - None: Load all 14 recordings. Default.

        cut_off_first_k_seconds: int
            The first seconds of a recording tend to be noisy and contain many outliers.
            Thus, we remove the first K seconds from each recording.

        n_max_minutes: int
            Use the first `n_max_minutes` of each recording.
            THIS VALUE IS NOT USED IN THIS CLASS!

        sfreq: int
            Re-sample the recording to this frequency.

        rng: int
            Random Seed

        do_common_average_reference: bool
            Defaults to False

        low_cut_hz: int, None
            Remove frequencies below this threshold. Defaults to None.

        high_cut_hz: int, None
            Remove frequencies above this threshold. Defaults to None.

        standardization_op: Text
            The standardization operation that is applied to each recording.
            Has to be one of exponential_moving_demean, exponential_moving_standardize, None.
            Defaults to exponential_moving_standardize. We have achieved the best performance
            with this operation.
        """

        # When the parameter `n_recordings_to_load is given as a integer, we interpret this value
        # as the subject id.
        if isinstance(n_recordings_to_load, int):
            n_recordings_to_load = [n_recordings_to_load]

        super(HighGammaData, self).__init__(
            data_path=data_path,
            n_recordings_to_load=n_recordings_to_load,
            n_max_minutes=n_max_minutes,
            cut_off_first_k_seconds=cut_off_first_k_seconds,
            sfreq=sfreq,
            rng=rng
        )

        # We apply a normalization operation to the split. This can have a significant impact on
        # the final performance. Using the `standardization` has shown the best results
        # in our experiments.
        _choices = ['exponential_moving_demean', 'exponential_moving_standardize', 'None', None]
        assert standardization_op in _choices, \
            'Parameter `standardization_op` unknown. ' \
            f'Has to be one of {_choices}, but was {standardization_op}'
        self.standardization_op = standardization_op

        if n_max_minutes is not None:
            logger.warning(
                'You have specified the value `n_max_minutes`, however, it is not used '
                'for this data set.'
            )

        # TODO: What impact has this parameter `do_common_average_reference`
        self.do_common_average_reference = do_common_average_reference

        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz

        # ------------------------- LOAD BASE DATA SET -------------------------------------------
        # We start to load the base data set by using the braindecode implementation.

        data_path = str(self.data_path)
        data_path += '/' if not data_path.endswith('/') else ''

        # TODO: Check this path here. Normally, already set in init.
        # os.environ['MNE_DATASETS_SCHIRRMEISTER2017_PATH'] = data_path

        logger.debug(
            f'Load base data from {os.environ["MNE_DATASETS_SCHIRRMEISTER2017_PATH"]} '
            f'with n_recordings_to_load={n_recordings_to_load}'
        )

        self.base_dataset = HGD(subject_ids=n_recordings_to_load)

        # ------------------------- CREATE SPLIT -------------------------------------------------
        # Then, we extract the correct split form the base data  set and apply the preprocessing.

        logger.debug('Original HGD dataset loaded. Start Creating Splits.')

        self.base_data_train, self.y_mean, self.y_std = self.load_split('train')
        self.base_data_eval, _, _ = self.load_split('eval')
        self.target_mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3}

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

        channel_names = [
            'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
            'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
            'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
            'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
            'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
            'CCP2h', 'CPP1h', 'CPP2h'
        ]
        channel_names = sorted(channel_names)
        return channel_names

    def load_split(self, split: str) -> Tuple[Any, float, float]:
        """
        Extract either the train or evaluation split from the High Gamma Data set.
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
        Mean    - Constant 0. Added to unify the interface.
        Std     - Constant 1.
        """

        assert split in ['train', 'eval'], 'Parameter `split` unknown. ' \
            f'Has to be one of [train, eval], but was {split}'

        # Since the braindecode version 0.5.1, we have to split along the column `train`.
        # This column has now values 'True', 'False' instead of 'train', 'eval'.
        split = 'True' if split == 'train' else 'False'

        # Extract the correct split from the base data set. We do this here subject-wise.
        whole_set = self.base_dataset.split(by='run')[split]

        preprocessors = self.get_preprocessing_pipeline()
        preprocess(whole_set, preprocessors)

        # This is a classification task -> targets are 0,.., 4 (representing the moved body part)
        y_mean, y_std = 0, 1

        logger.debug(f'Dataset: {split}. Y_mean = {y_mean}. Y_std = {y_std}')
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
                # In comparison to the TUH data set, we use the entire duration starting at tmin.
                tmax=None,
                include_tmax=True,
                apply_on_array=False
            ),

            # Convert from volt to microvolt
            Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),

            # Clip the signal between -800 and 800 microVolt
            Preprocessor(fn=lambda x: np.clip(x, -800, 800), apply_on_array=True),
        ]

        if self.do_common_average_reference:
            preprocessors.append(Preprocessor(fn='set_eeg_reference', ref_channels='average'))

        preprocessors += [
            # Resample the signal to have a frequency of `sfreq`
            Preprocessor(fn='resample', sfreq=self.sfreq, apply_on_array=False),

            # Remove freqencies that are not between `low_cut_hz` and higher than `high_cut_hz`
            Preprocessor(
                fn='filter',
                l_freq=self.low_cut_hz,
                h_freq=self.high_cut_hz,
                apply_on_array=False
            )
        ]

        # We standardize the split using a demean or a 0-mean-unit-variance standardization.
        # But we also allow to have no standardization at all.
        if not self.standardization_op in ['None', None]:

            preprocessors.append(
                Preprocessor(
                    fn=NORMALIZATION_FUNCS[self.standardization_op],
                    init_block_size=1000, factor_new=1e-3,
                    apply_on_array=True
                )
            )

        return preprocessors


class HighGammaDataAllChannels(HighGammaData):
    """
    The High Gamma Data Set contains also more channels. This version uses all available sensors.
    In our experiments, we did not use this version. We have added this class just for reasons
    of completeness.
    """
    @staticmethod
    def get_ordered_channel_names() -> List[str]:
        """
        Each channel has a unique name. These channel names correspond to the sensors used
        during the recording.

        Returns
        -------
        List
        """

        channel_names = [
            'Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2',
            'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1',
            'Oz', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3',
            'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', 'P1',
            'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8',
            'PO7', 'PO8', 'FT9', 'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9',
            'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
            'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
            'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h',
            'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h',
            'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h',
            'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h',
            'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h'
        ]
        channel_names = sorted(channel_names)
        return channel_names
