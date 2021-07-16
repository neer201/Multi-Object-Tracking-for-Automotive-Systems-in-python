import cProfile
import pstats
from io import StringIO


class Profiler(object):
    def __init__(
        self,
        enabled=False,
        contextstr=None,
        fraction=1.0,
        sort_by="cumulative",
        parent=None,
        logger=None,
    ):
        self.enabled = enabled

        self.contextstr = contextstr or str(self.__class__)

        if fraction > 1.0 or fraction < 0.0:
            fraction = 1.0

        self.fraction = fraction
        self.sort_by = sort_by

        self.parent = parent
        self.logger = logger

        self.stream = StringIO()
        self.profiler = cProfile.Profile()

    def __enter__(self, *args):

        if not self.enabled:
            return self

        # Start profiling.
        self.stream.write("\nprofile: {}: enter\n".format(self.contextstr))
        self.profiler.enable()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if not self.enabled:
            return False

        self.profiler.disable()

        sort_by = self.sort_by
        ps = pstats.Stats(self.profiler, stream=self.stream).sort_stats(sort_by)
        ps.print_stats(self.fraction)

        self.stream.write("\nprofile: {}: exit\n".format(self.contextstr))

        return False

    def get_profile_data(self):

        value = self.stream.getvalue()
        if self.logger is not None:
            self.logger.info("%s", value)

        return value


if __name__ == "__main__":  # pragma: no cover

    import re

    with Profiler(enabled=True, contextstr="test") as p:
        for _i in range(1000):
            r = re.compile(r"^$")

    print(p.get_profile_data())

    try:
        with Profiler(enabled=True, contextstr="exception") as p:
            raise ValueError("Error")
    except ValueError:
        print(p.get_profile_data())

    profiling_enabled = False
    with Profiler(enabled=profiling_enabled, contextstr="not enabled") as p:
        for _i in range(1000):
            r = re.compile(r"^$")

    print(p.get_profile_data())
