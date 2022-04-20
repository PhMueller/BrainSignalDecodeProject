import torch
from torch import Tensor

from braindecode.augmentation.transforms import Transform as BrainDecodeTransform, Output


def mixup(X, y, lam, idx_perm):
    """Mixes two channels of EEG data.

    See [1]_ for details.
    Implementation based on [2]_.

    NOTE: We have changed two lines to add the support for multiclass labels.

    Parameters
    ----------
    X : torch.Tensor
        EEG data in form ``batch_size, n_channels, n_times``
    y : torch.Tensor
        Target of length ``batch_size``
    lam : torch.Tensor
        Values between 0 and 1 setting the linear interpolation between
        examples.
    idx_perm: torch.Tensor
        Permuted indices of example that are mixed into original examples.

    Returns
    -------
    tuple
        ``X``, ``y``. Where ``X`` is augmented and ``y`` is a tuple  of length
        3 containing the labels of the two mixed channels and the mixing
        coefficient.

    References
    ----------
    .. [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        (2018). mixup: Beyond Empirical Risk Minimization. In 2018
        International Conference on Learning Representations (ICLR)
        Online: https://arxiv.org/abs/1710.09412
    .. [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
     """
    device = X.device
    batch_size, n_channels, n_times = X.shape

    X_mix = torch.zeros((batch_size, n_channels, n_times)).to(device)

    # We changed the following two lines to add the support for multiclass labels.
    y_a = torch.zeros_like(y).to(device)
    y_b = torch.zeros_like(y).to(device)

    for idx in range(batch_size):
        X_mix[idx] = lam[idx] * X[idx] \
            + (1 - lam[idx]) * X[idx_perm[idx]]
        y_a[idx] = y[idx]
        y_b[idx] = y[idx_perm[idx]]

    return X_mix, (y_a, y_b, lam)


class Mixup(BrainDecodeTransform):
    """Implements Iterator for Mixup for EEG data. See [1]_.
    Implementation based on [2]_.

    Parameters
    ----------
    alpha: float
        Mixup hyperparameter.
    beta_per_sample: bool (default=False)
        By default, one mixing coefficient per batch is drawn from a beta
        distribution. If True, one mixing coefficient per sample is drawn.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
       (2018). mixup: Beyond Empirical Risk Minimization. In 2018
       International Conference on Learning Representations (ICLR)
       Online: https://arxiv.org/abs/1710.09412
    .. [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    operation = staticmethod(mixup)

    def __init__(
        self,
        alpha,
        beta_per_sample=False,
        random_state=None
    ):
        super().__init__(
            probability=1.0,  # Mixup has to be applied to whole batches
            random_state=random_state
        )
        self.alpha = alpha
        self.beta_per_sample = beta_per_sample

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params: dict
            Contains the values sampled uniformly between 0 and 1 setting the
            linear interpolation between examples (lam) and the shuffled
            indices of examples that are mixed into original examples
            (idx_perm).
        """
        X = batch[0]
        device = X.device
        batch_size, _, _ = X.shape

        if self.alpha > 0:
            if self.beta_per_sample:
                lam = torch.as_tensor(
                    self.rng.beta(self.alpha, self.alpha, batch_size)
                ).to(device)
            else:
                lam = torch.ones(batch_size).to(device)
                lam *= self.rng.beta(self.alpha, self.alpha)
        else:
            lam = torch.ones(batch_size).to(device)

        idx_perm = torch.as_tensor(self.rng.permutation(batch_size,))

        return {
            "lam": lam,
            "idx_perm": idx_perm,
        }

    def forward(self, X: Tensor, y: Tensor = None) -> Output:
        x_mixed, (y_a, y_b, mixing_coef) = super(Mixup, self).forward(X, y)

        # y has either a shape of
        # (Batch), (Batch x 1), (Batch, 1), (Batch x 1 x N_classes). (the mixing coef have always shape (Batch))
        y_a = y_a.squeeze()
        y_b = y_b.squeeze()

        if y_a.dim() > 1:
            # if the mixed target vectors are 2d, we need to add a singleton dim to the coef, so that the multiplication
            # is correct.
            mixing_coef = mixing_coef.unsqueeze(1)

        # Interpolate between the mixed target vectors.
        y_mixed = y_a * mixing_coef + y_b * (1 - mixing_coef)
        y_mixed = y_mixed.view_as(y)

        return x_mixed, y_mixed
