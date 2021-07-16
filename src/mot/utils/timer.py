import logging
import time
from functools import wraps


class Timer:
    def __init__(self, logger=logging.debug, time_source=time.perf_counter, name=None):
        self.logger = logger
        self.time_source = time_source
        self.name = name
        self.start, self.stop = None, None

    def __enter__(self):
        self.start = self.time_source()

    def __exit__(self, *exc_info):
        self.stop = self.time_source()
        self.logger(f"{self.name} {self.duration():.3f} s")

    def duration(self):
        return self.stop - self.start

    def __call__(self, method):
        @wraps(method)
        def _whraped_method(self, *method_args, **method_kwargs):
            with Timer(
                logger=logging.debug,
                time_source=time.time,
                name=method.__name__,
            ):
                method_output = method(self, *method_args, **method_kwargs)
            return method_output

        return _whraped_method
