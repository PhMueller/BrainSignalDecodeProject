from .age_benchmark import AgeBaselineBenchmark, AgeBenchmark

BASE_DEBUG_SETTINGS = {
    'print_step': 1,
    'print_epoch': 10,
    'val_minutes': 0.5,
    'limit_train_batches': 2,
    'limit_val_batches': 2
}

TUH_DEBUG_SETTINGS = {
    'n_recordings_to_load': 300,
    **BASE_DEBUG_SETTINGS
}

HGD_DEBUG_SETTINGS = {
    'n_recordings_to_load': 1,
    **BASE_DEBUG_SETTINGS
}


__all__ = ['AgeBenchmark', 'AgeBaselineBenchmark']
