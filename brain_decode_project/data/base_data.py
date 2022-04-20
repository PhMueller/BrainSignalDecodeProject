from abc import ABC
from pathlib import Path
from typing import Union, Tuple, Any, List, Iterable, Dict

from braindecode.preprocessing import exponential_moving_demean, exponential_moving_standardize
from braindecode.datasets.base import WindowsDataset


class BaseData(ABC):
    """
    This is the parent class for every EEG Data set. We use this class to load the corresponding
    data set from the braindecode project.

    This class implements the interface for loading the data.

    """
    def __init__(
            self,
            data_path: Union[str, Path],
            n_recordings_to_load: Union[int, Iterable, None],
            cut_off_first_k_seconds: int,
            n_max_minutes: int,
            sfreq: int,
            rng: int,
    ):
        # We instantiate the data sets from the brain decode package.
        self.base_dataset = None

        # After that we split the data into train and evaluation. `base_data_train` is
        # potentially further split into train and validation during HPO.
        self.base_data_train = None

        # This one only contains the test split
        self.base_data_eval = None

        # Mean and std across the train targets
        self.y_mean = None
        self.y_std = None

        # Target mapping is needed for creating the windows.
        self.target_mapping = None

        self.data_path = data_path

        # This is the number of recording we load from the data set.
        self.n_recordings_to_load = range(n_recordings_to_load) \
            if isinstance(n_recordings_to_load, int) \
            else n_recordings_to_load

        self.n_max_minutes = n_max_minutes

        self.cut_off_first_k_seconds = cut_off_first_k_seconds

        self.sfreq = sfreq

        self.rng = rng

        # The channel list corresponds to the dimensionality of the data. Each row corresponds to
        # a named channel recorded with the sensor cap. These information are important for later
        # augmentations.
        self.ch_names = self.get_ordered_channel_names()
        self.num_channels = len(self.ch_names)

    def load_split(self, split: str) -> Tuple[Any, float, float]:
        """
        Base implementation: Extract either the train or evaluation split from the base data set.
        This function also applies a list of preprocessing operations.

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
        raise NotImplementedError()

    def get_preprocessing_pipeline(self, **kwargs: Dict) -> List:
        raise NotImplementedError()

    @staticmethod
    def get_ordered_channel_names() -> List[str]:
        """ Get the list of ordered channel names.

        This list corresponds to the channels in the data matrix.
        They describe the signals observed by the different sensors.
        """
        raise NotImplementedError


class BaseDataSplitter(ABC):
    """
    This is the parent class for every EEG Data set. We use this class to load the corresponding
    data set from the braindecode project.

    This class implements the interface for loading the data.

    """
    def __init__(
            self,
            input_window_samples: int,
            window_stride_samples: int,
    ):
        self.input_window_samples = input_window_samples
        self.window_stride_samples = window_stride_samples

    def split_into_train_valid(
        self,
        dataset,
    ) -> Tuple[WindowsDataset, WindowsDataset]:

        raise NotImplementedError()

    def split_into_final_train_valid(
            self,
            dataset,
    ) -> Tuple[WindowsDataset, WindowsDataset]:

        raise NotImplementedError()


NORMALIZATION_FUNCS = {
    'exponential_moving_demean': exponential_moving_demean,  # Default
    'exponential_moving_standardize': exponential_moving_standardize,
}