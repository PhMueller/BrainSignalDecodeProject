import os


def remove_slurm_env_variables():
    """
    Remove the slurm related environment variables. Pytorch lightning looks for them and
    if it finds them, it starts some crazy slurm functions.
    However, this crashes because it is not called in the main process.
    Easy fix: Remove the variables, simulate a local run.
    """
    if "SLURM_NTASKS" in os.environ:
        del os.environ["SLURM_NTASKS"]
    if "SLURM_JOB_NAME" in os.environ:
        del os.environ["SLURM_JOB_NAME"]


def set_required_env_variables():
    """
    Running on the slurm cluster requires to set some environment variables.
    """
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_MAX_THREADS'] = '1'
    os.environ['MKL_DISABLE_FAST_MM'] = '1'
