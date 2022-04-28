import logging


def setup_logger_for_objective_function(debug: bool = False) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    main_logger = logging.getLogger('DEHB-Main')
    main_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # This logger prints tons of line during the loading of the data -> Unnecessary: We disable it
    mne_logger = logging.getLogger('mne')
    mne_logger.setLevel(logging.WARNING)