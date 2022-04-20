"""
DATA
====

This directory contains all classes that are associated to the data sets.

In this project, we have used two different data sets.

1) TUH Abnormal EEG:
--------------------
We use the 2.0.0 Version of the TUH Abnormal EEG data set.

We derive three possible targets from
this data set:
- AGE:          The chronological age of a subject (Regression)
- GENDER:       The gender of a subject (Binary Classification)
- PATHOLOGICAL: The presence of a illness (Binary Classification)

You can download the data set from:
https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab

2) High Gamma Data Set:
-----------------------
This data set is from "Deep learning with convolutional neural networks for
EEG decoding and visualization" (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730).

We derive a single task form this data set:
HGD:            Classification of a subjects movement (4 possibilities)

You can download the data set from:
https://github.com/robintibor/high-gamma-dataset
"""

import logging

# The recordings in the data sets are created using the `mne` package. This generates many - for
# our case redundant - log messages. We disable the logger.
mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.WARNING)

from brain_decode_project.data.hgd_data import HighGammaData, HighGammaDataAllChannels
from brain_decode_project.data.tuh_data import TUHData

__all__ = ['TUHData', 'HighGammaData', 'HighGammaDataAllChannels']
