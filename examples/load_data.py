"""
This script shows how the data is loaded and preprocessed.
"""

from pathlib import Path
from torch.utils.data import DataLoader

from brain_decode_project.data import TUHData, TUHDataSplitter

# -------------------------- SETTINGS ------------------------------------------------------------
data_path = Path('/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/')
n_recordings_to_load = 300  # 2993: entire data set

data_target_name = 'age'

# The next two parameters change the data set. It means we take only observations from
# [(<cut_off_first_k_seconds> * 60):<n_max_minutes>] ( here: from 3 to 5 minutes. )
cut_off_first_k_seconds = 3 * 60
n_max_minutes = 5 * 60
sfreq = 100
rng = 0

# Option to load either only train or test set or both
train_or_eval = 'both'  # train, eval, both

# If set, only healthy patients are included in the data set
only_healthy = False

# Either exponential_moving demean or exponential_moving_standardize. Has a big impact on final
# peformance. I have selected here demean for debugging reasons.
standardization_op = 'exponential_moving_demean'

# Number of predictions, returned from network
window_stride_samples = 1104
input_window_samples = 1600

batch_size = 32
# -------------------------- SETTINGS ------------------------------------------------------------

# -------------------------- RUN -----------------------------------------------------------------
# Data loading and preprocessing happens in this step.
dataset = TUHData(
    data_path=data_path,
    n_recordings_to_load=n_recordings_to_load,
    data_target_name=data_target_name,
    cut_off_first_k_seconds=cut_off_first_k_seconds,
    n_max_minutes=n_max_minutes,
    sfreq=sfreq,
    rng=rng,
    train_or_eval=train_or_eval,
    only_healthy=only_healthy,
    standardization_op=standardization_op,
)

# We create windows from every trail and then split the data into 80 train and 20 validation set.
dataset_splitter = TUHDataSplitter(input_window_samples=input_window_samples,
                                   window_stride_samples=window_stride_samples)
train_set, valid_set = dataset_splitter.split_into_train_valid(dataset)

train_dl = DataLoader(train_set, batch_size=batch_size, drop_last=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, drop_last=False)

print('Done')
