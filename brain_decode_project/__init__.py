import logging
import os
from pathlib import Path

__contact__ = "muelleph@cs.uni-freiburg.de"

# We initialize the root logger with a consistent format.
_default_log_format = '[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'
logging.basicConfig(format=_default_log_format, level=logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

_root_handler = root_logger.handlers[0]
_format = logging.Formatter(_default_log_format)
_root_handler.setFormatter(_format)

# Some constants I often use. Feel free to adapt.
DATA_PATH_LOCAL = Path('/media/philipp/Volume/Code/EEG_Data/tuh_eeg_abnormal/')
DATA_PATH_CLUSTER = Path('/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/')
RESULT_PATH_LOCAL = Path('/media/philipp/Volume/Code/EEG_Data/results/')
RESULT_PATH_CLUSTER = Path('/work/dlclarge1/muelleph-EEG_Project/results/')

# We work on a SLURM cluster. When the job is scheduled there is a environment variable called
# SLURM_JOBID. When this flag is set, we use a different printing mechanism, e.g. do not show a
# progressbar.
ON_CLUSTER = os.environ.get('SLURM_JOBID') is not None
SHOW_PROGRESSBAR = not ON_CLUSTER

DEBUG_SETTINGS = {
    'n_recordings_to_load': 300,
    'print_step': 1,
    'print_epoch': 1,
    'val_minutes': 0.5,
    'limit_train_batches': 5,
    'limit_val_batches': 5
}

DATA_PATH_HIGH_GAMMA_LOCAL = DATA_PATH_LOCAL.parent / 'Schirrmeister2017'
DATA_PATH_HIGH_GAMMA_CLUSTER = Path('/work/dlclarge1/muelleph-EEG_Project/data/Schirrmeister2017')
os.environ['MNE_DATASETS_SCHIRRMEISTER2017_PATH'] = str(DATA_PATH_HIGH_GAMMA_CLUSTER) \
    if ON_CLUSTER else str(DATA_PATH_HIGH_GAMMA_LOCAL)
