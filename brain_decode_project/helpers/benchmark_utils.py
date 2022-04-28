import logging
from brain_decode_project.benchmarks.compound_benchmark import ComposedBenchmark
from typing import Type


def get_available_benchmarks():
    import inspect
    import brain_decode_project.benchmarks
    cls_members = inspect.getmembers(brain_decode_project.benchmarks, inspect.isclass)
    return [cls[0] for cls in cls_members]


def is_benchmark_available(benchmark_name):
    cls_names = get_available_benchmarks()
    is_available = benchmark_name in cls_names
    if not is_available:
        logging.info(f'{benchmark_name} not in {cls_names}')
    return is_available


def get_benchmark_object(benchmark_name) -> ComposedBenchmark:
    """
    Helperfunction: Returns the uninstantiated benchmark class.
    Example:
    > benchmark_object = get_benchmark_object('AgeBenchmarkBaseline')
    > benchmark = benchmark_object(run_results='.', data_dir='.')

    Parameters
    ----------
    benchmark_name: str
        Name of a benchmark. Check 'brain_decode_project.benchmarks.__init__' for available
        options.

    Returns
    -------
    Type[ComposedBenchmark]
    """
    module = __import__('brain_decode_project.benchmarks', fromlist=[benchmark_name])
    benchmark_obj = getattr(module, benchmark_name)
    return benchmark_obj
