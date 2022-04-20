import torch
from mne.channels import make_standard_montage

import numpy as np
from typing import Union, List, Tuple

from braindecode.augmentation import TimeReverse, SignFlip, FTSurrogate, ChannelsShuffle,\
    ChannelsDropout, GaussianNoise, ChannelsSymmetry, SmoothTimeMask, BandstopFilter,\
    FrequencyShift, SensorsRotation, SensorsZRotation, SensorsYRotation, SensorsXRotation
from brain_decode_project.networks.mixup import Mixup


class AugmentationBox(torch.nn.Module):
    """
    Collection of all augmentation operations.
    """

    def __init__(
            self,
            p_time_reverse: float = 0.0,
            p_sign_flip: float = 0.0,
            p_ft_surrogate: float = 0.0,
            p_channel_shuffle: float = 0.0,
            p_channel_dropout: float = 0.0,
            p_gaussian_noise: float = 0.0,
            p_channel_symmetry: float = 0.0,
            p_smooth_time_mask: float = 0.0,
            p_mixup: float = 0.0,
            p_frequency_shift: float = 0.0,
            # p_bandstop_filter: float = 0.0,
            p_sensor_rotation: float = 0.0,
            seed: Union[int, None] = None,
            sfreq: int = None,
            ordered_channel_names: List[str] = None,
    ):

        super(AugmentationBox, self).__init__()

        self.random_state = np.random.RandomState(seed=seed)
        self.sfreq = sfreq
        self.ordered_channel_names = ordered_channel_names
        self.ordered_channel_positions = self._get_positions(self.ordered_channel_names)

        self.time_reverse = TimeReverse(probability=p_time_reverse, random_state=self.random_state)
        self.sign_flip = SignFlip(probability=p_sign_flip, random_state=self.random_state)

        self.ft_surrogate = FTSurrogate(
            probability=p_ft_surrogate, phase_noise_magnitude=1, random_state=self.random_state
        )

        self.gaussian_noise = GaussianNoise(
            probability=p_gaussian_noise, std=0.1, random_state=self.random_state
        )

        self.channels_symmetry = ChannelsSymmetry(
            probability=p_channel_symmetry, ordered_ch_names=self.ordered_channel_names,
            random_state=self.random_state
        )
        self.channels_shuffle = ChannelsShuffle(
            probability=p_channel_shuffle, p_shuffle=0.2, random_state=self.random_state
        )

        # Drop multiple channels randomly. (each channel with 0.2)
        self.channels_dropout = ChannelsDropout(
            probability=p_channel_dropout, p_drop=0.2, random_state=self.random_state
        )
        # Smoothly replace a randomly chosen contiguous part of all channels by zeros
        self.smooth_time_mask = SmoothTimeMask(
            probability=p_smooth_time_mask, mask_len_samples=self.sfreq,
            random_state=self.random_state
        )

        # TODO: Find a good explanation for this augmentation.
        # self.bandstop_filter = BandstopFilter(
        #     probability=p_bandstop_filter, sfreq=self.sfreq / 2,
        #     random_state=self.random_state
        # )

        # Does this work and how?
        self.frequency_shift = FrequencyShift(
            probability=p_frequency_shift, sfreq=self.sfreq, max_delta_freq=2,
            random_state=self.random_state
        )

        # TODO: I think the current code allows only a single rotation. It does not return the
        # updated positions after performing a rotation.
        self.sensor_rotation = SensorsRotation(
            probability=p_sensor_rotation, axis='z', max_degrees=15,
            sensors_positions_matrix=self.ordered_channel_positions,
            random_state=self.random_state
        )
        # SensorsZRotation(probability=p_sensor_rotation_z, sensors_positions_matrix=self.ordered_channel_positions, random_state=self.random_state),
        # SensorsYRotation(probability=p_sensor_rotation_y, sensors_positions_matrix=self.ordered_channel_positions, random_state=self.random_state),
        # SensorsXRotation(probability=p_sensor_rotation_x, sensors_positions_matrix=self.ordered_channel_positions, random_state=self.random_state),

        # Only for classification tasks
        self.p_mixup = p_mixup
        self.mixup = Mixup(alpha=0.5, random_state=self.random_state)  # noqa

    def forward(self, inputs, targets) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs, targets = self.time_reverse(inputs, targets)
        inputs, targets = self.sign_flip(inputs, targets)
        inputs, targets = self.ft_surrogate(inputs, targets)
        inputs, targets = self.gaussian_noise(inputs, targets)
        inputs, targets = self.channels_symmetry(inputs, targets)
        inputs, targets = self.channels_shuffle(inputs, targets)
        inputs, targets = self.channels_dropout(inputs, targets)
        inputs, targets = self.smooth_time_mask(inputs, targets)
        # inputs, targets = self.bandstop_filter(inputs, targets)
        inputs, targets = self.frequency_shift(inputs, targets)
        inputs, targets = self.sensor_rotation(inputs, targets)

        if self.random_state.random() <= self.p_mixup:
            # In contrast to the other augmentations, this operation (if active) is always
            # applied to the entire batch. `p_mixup` describes if it is active or not.
            inputs, targets = self.mixup(inputs, targets)

        return inputs, targets

    @staticmethod
    def get_hp_names() -> List[str]:
        return [
            'p_time_reverse', 'p_sign_flip', 'p_ft_surrogate', 'p_channel_shuffle',
            'p_channel_dropout', 'p_gaussian_noise', 'p_channel_symmetry', 'p_smooth_time_mask',
            'p_mixup', 'p_frequency_shift',
            # 'p_bandstop_filter',
            'p_sensor_rotation',
        ]

    @staticmethod
    def _get_positions(channel_names: List[str]) -> torch.Tensor:
        """
        Look up the positions as 3D coordinates for a list of channel names.
        Either use the 1005 for the High Gamma Data or the 1020 for the TUH Abnormal data set.

        Note:
        -----
            Even if 1020 is a subset of 1005, the positions differ by a small value.
            We need to assume that it is therefore crucial which mapping to select.

        Parameters
        ----------
        channel_names: List[str]
            Channel names used for the data. E.g. ['EEG A1-REF', 'EEG FP1-REF',... ]

        Returns
        -------
        np.ndarray with shape 3 x len(channel_names)

        """
        # Some names are in Camelcase while our channel names are all in uppercase
        # -> replace the keys with their upper case version.
        mapping_1020 = make_standard_montage('standard_1020').get_positions()['ch_pos']
        mapping_1020 = {key.upper(): value for key, value in mapping_1020.items()}

        mapping_1005 = make_standard_montage('standard_1005').get_positions()['ch_pos']
        mapping_1005 = {key.upper(): value for key, value in mapping_1005.items()}

        # The TUH channels are named as follows: "EEG <ChannelName>-REF".
        # But in the mapping, they are only described with <ChannelName>.
        channel_names = [c.lstrip('EEG ').rstrip('-REF')
                         if c.startswith('EEG ') else c for c in channel_names]

        use_1020 = all((c.upper() in mapping_1020 for c in channel_names))
        use_1005 = all((c.upper() in mapping_1005 for c in channel_names))

        assert use_1005 or use_1020, 'Didn\'t find all channels in the 1020 or 1005 mapping.'

        mapping = mapping_1020 if use_1020 else mapping_1005

        positions = []
        for channel in channel_names:
            # Our channels are named "EEG <ChannelName>-REF".
            # In the mapping they are declared only with <ChannelName>.
            positions.append(mapping[channel.upper()])

        positions = np.array(positions)
        positions = torch.as_tensor(positions).T
        return positions


if __name__ == '__main__':

    x = torch.rand((10, 21, 1600))
    y = torch.arange(10).unsqueeze(1).unsqueeze(1) # regression
    y = y * torch.ones((10, 1, 3))  # classification
    # y = torch.arange(10)

    a = AugmentationBox(
        p_time_reverse=0.0,
        p_sign_flip=0.0,
        p_ft_surrogate=0.0,
        p_channel_shuffle=0.0,
        p_channel_dropout=0.0,
        p_gaussian_noise=0.0,
        p_channel_symmetry=0.0,
        p_smooth_time_mask=0.0,
        p_mixup=1.0,
        p_frequency_shift=0.0,
        # p_bandstop_filter=0.0,
        p_sensor_rotation=0.0,
        seed=0
    )

    out_x, out_y = a(x, y)
