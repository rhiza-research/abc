# General utility functions for all parts of the pipeline
import datetime
import os
import pathlib
import shutil as sh
import subprocess
import sys
import time
import warnings

import requests
import ssl
from urllib3 import poolmanager

DATETIME_FORMAT = "%Y%m%d"


class TLSAdapter(requests.adapters.HTTPAdapter):

    def init_poolmanager(self, connections, maxsize, block=False):
        """Create and initialize the urllib3 PoolManager."""
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        self.poolmanager = poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_context=ctx)


def download_url(url, timeout=600, retry=3, cookies={}):
    """Download URL, waiting some time between retries."""
    r = None
    session = requests.session()
    session.mount('https://', TLSAdapter())

    for i in range(retry):
        try:
            r = requests.get(url, timeout=timeout, cookies=cookies)
            r = session.get(url, timeout=timeout, cookies=cookies)
            return r
        except requests.exceptions.Timeout as e:
            # Wait until making another request
            if i == retry - 1:
                raise e
            print(f"Request to url {url} has timed out. Trying again...")
            time.sleep(3)
    print(f"Failed to retrieve file after {retry} attempts. Stopping...")


def get_dates(date_str):
    """Outputs the list of dates corresponding to input date string."""
    if "-" in date_str:
        # Input is of the form '20170101-20180130'
        first_date, last_date = date_str.split("-")
        first_date = string_to_dt(first_date)
        last_date = string_to_dt(last_date)
        dates = [first_date + datetime.timedelta(days=x)
                 for x in range(0, (last_date - first_date).days + 1)]
        return dates
    elif "," in date_str:
        # Input is of the form '20170101,20170102,20180309'
        dates = [datetime.datetime.strptime(
            x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif "," in date_str:
        # Input is of the form '20170101,20170102,20180309'
        dates = [datetime.datetime.strptime(
            x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif len(date_str) == 4:
        # Input '2017' is expanded to 20170101-20171231
        year = int(date_str)
        first_date = datetime.datetime(year=year, month=1, day=1)
        last_date = datetime.datetime(year=year, month=12, day=31)
        dates = [first_date + datetime.timedelta(days=x)
                 for x in range(0, (last_date - first_date).days)]
        return dates
    elif len(date_str) == 6:
        # Input '201701' is expanded to 20170101-20170131
        year = int(date_str[0:4])
        month = int(date_str[4:6])

        first_date = datetime.datetime(year=year, month=month, day=1)
        if month == 12:
            last_date = datetime.datetime(year=year + 1, month=1, day=1)
        else:
            last_date = datetime.datetime(year=year, month=month + 1, day=1)
        dates = [first_date + datetime.timedelta(days=x)
                 for x in range(0, (last_date - first_date).days)]
        return dates
    elif len(date_str) == 8:
        # Input '20170101' is a date
        dates = [datetime.datetime.strptime(date_str.strip(), "%Y%m%d")]
        return dates
    else:
        raise NotImplementedError(
            "Date string provided cannot be transformed " "into list of target dates.")


def printf(str):
    """Calls print on given argument and then flushes
    stdout buffer to ensure printed message is displayed right away
    """
    print(str, flush=True)


def print_fail(message="FAIL", verbose=True, skip_line_before=True, skip_line_after=True, bold=False):
    """Print message in purple."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(
                f"{string_before}\x1b[1;30;45m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[35m{message}\x1b[0m{string_after}")


def print_error(message="ERROR", verbose=True, skip_line_before=True, skip_line_after=True, bold=False):
    """Print message in red."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(
                f"{string_before}\x1b[1;30;41m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[31m{message}\x1b[0m{string_after}")


def print_warning(message="WARNING", verbose=True, skip_line_before=True, skip_line_after=True, bold=False):
    """Print message in yellow."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(
                f"{string_before}\x1b[1;30;43m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[33m{message}\x1b[0m{string_after}")


def print_ok(message="OK", verbose=True, skip_line_before=True, skip_line_after=True, bold=False):
    """Print message in green."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(
                f"{string_before}\x1b[1;30;42m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[32m{message}\x1b[0m{string_after}")


def print_info(message, verbose=True):
    if verbose:
        print(message)


def make_directories(dirname):
    """Creates directory and parent directories with 777 permissions
    if they do not exist
    """
    if dirname != "":
        os.umask(0)
        os.makedirs(dirname, exist_ok=True, mode=0o777)


def make_parent_directories(file_path):
    """Creates parent directories of a given file with 777 permissions
    if they do not exist
    """
    make_directories(os.path.dirname(file_path))


def symlink(src, dest, use_abs_path=False):
    """Symlinks dest to point to src; if dest was previously a symlink,
    unlinks src first

    Args:
      src - source file name
      dest - target file name
      use_abs_path - if True, links dest to the absolute path of src
        (useful when src is not expressed relative to dest)
    """
    # n and f flags ensure that prior symlink is overwritten by new one
    if use_abs_path:
        src = os.path.abspath(src)
    cmd = "ln -nsf {} {}".format(src, dest)
    subprocess.call(cmd, shell=True)


def get_folder(folder_path, verbose=True):
    """Creates folder, if it doesn't exist, and returns folder path.
    Args:
        folder_path (str): Folder path, either existing or to be created.
    Returns:
        str: folder path.
    """
    folder_path = pathlib.Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path


def set_path_permission(file_path, mode=0o777, recursive=True, warning=False):
    if recursive:
        recursive_path = ""
        for dir in os.path.normpath(file_path).split(os.path.sep):
            recursive_path = os.path.join(recursive_path, dir)
            set_path_permission(recursive_path, mode=mode,
                                recursive=False, warning=warning)
    else:
        try:
            os.chmod(file_path, mode)
            sh.chown(file_path, group="sched_mit_hill")
        except PermissionError:
            if warning:
                print_warning(
                    f"Permission error: can't modify path ({file_path})", skip_line_before=False, skip_line_after=False
                )
            pass


def set_file_permissions(file_path, skip_if_exists=False, throw=False, mode=0o777):
    """Set file/folder permissions.

    Parameters
    ----------
    skip_if_exists : boolean
        If True, skips setting permissions if file exists

    throw : boolean
        If True, throws exception if cannot set permissions

    """
    if not skip_if_exists or (skip_if_exists and not os.path.exists(file_path)):
        # Set permissions
        try:
            os.chmod(file_path, mode)
            sh.chown(file_path, group="sched_mit_hill")
        except Exception as err:
            if throw:
                print_warning("Cannot modify file permissions.")
                raise err
            else:
                pass


def get_task_from_string(task_str):
    """
    Gets a region, gt_id, horizon from a task string. Returns None if invalid
    task string
    Args:
        task_str: string in format "<region>_<gt_id>_<horzion>
    """
    try:
        region, gt_id, horizon = task_str.split("_")
        if region not in ["contest", "us"]:
            raise ValueError("Bad region.")

        if gt_id not in ["tmp2m", "precip"]:
            raise ValueError("Bad gt_id.")

        if horizon not in ["12w", "34w", "56w"]:
            raise ValueError("Bad horizon.")

    except Exception:
        printf("Could not get task parameters from task string.")
        return None

    return region, gt_id, horizon


def num_available_cpus():
    """Returns the number of CPUs available considering the sched_setaffinity
    Linux system call, which limits which CPUs a process and its children
    can run on.
    """
    return len(os.sched_getaffinity(0))


def hash_strings(strings, sort_first=True):
    """Returns a string hash value for a given list of strings.
    Always returns the same value for the same inputs.

    Args:
      strings: list of strings to hash
      sort_first: sort string list before hashing? if True, returns the same
        hash for the same collection of strings irrespective of their ordering
    """
    if sort_first:
        strings = sorted(strings)
    # Setting environment variable PYTHONHASHSEED to 0 disables hash randomness
    # Must be done prior to program execution, so we call out to a new Python
    # process
    return subprocess.check_output(
        "export PYTHONHASHSEED=0 && python -c \"print(str(abs(hash('{}'))))\"".format(
            ",".join(strings)),
        shell=True,
        universal_newlines=True,
    ).strip()


def string_to_dt(string):
    """Transforms string to datetime."""
    return datetime.datetime.strptime(string, DATETIME_FORMAT)


def dt_to_string(dt):
    """Transforms datetime to string."""
    return datetime.datetime.strftime(dt, DATETIME_FORMAT)


def get_dt_range(base_date, days_ahead_start=0, days_ahead_end=0):
    """Lists the dates between (base_date + days_ahead_start), included, and
    (base_date + days_ahead_end), not included.

    Parameters
    ----------
    base_date : datetime
        Reference date for time window.
    days_ahead_start : int
        Time window starts days_ahead_start days from base_date (included).
    days_ahead_end : int
        Time window ends days_ahead_start days from base_date (included).

    Returns
    -------
    List
        Dates in (base_date + days_ahead_start) and (base + days_ahead_end)

    """
    date_start = base_date + datetime.timedelta(days=days_ahead_start)
    days_in_window = days_ahead_end - days_ahead_start

    date_list = [date_start +
                 datetime.timedelta(days=day) for day in range(days_in_window)]
    return date_list


def get_current_year():
    """Gets year at the time when the script is run."""
    now = datetime.datetime.now()
    return now.year


class TicToc(object):
    """
    Author: Hector Sanchez
    Date: 2018-07-26
    Description: Class that allows you to do 'tic toc' to your code.

    This class was based the answers that you can find in the next url.
    https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

    How to use it:

    with TicToc('name'):
      some code....

    or

    t = TicToc('name')
    t.tic()
    some code...
    t.toc()
    print(t.elapsed)

    or

    t = TicToc('name',time.clock) # or any other method.
                                 # time.clock seems to be deprecated
    with t:
      some code....

    or

    t = TicToc()
    t.tic()
    t.tic()
    t.tic()
    t.toc()
    t.toc()
    t.toc()
    print(t.elapsed)

    or

    from src.utils.tictoc import tic,toc

    tic()
    tic()
    toc()
    toc()
    """

    def __init__(self, name="", method="time", nested=False, print_toc=True):
        """
        Args:
        name (str): Just informative, not needed
        method (int|str|ftn|clss): Still trying to understand the default
            options. 'time' uses the 'real wold' clock, while the other
            two use the cpu clock. If you want to use your own method, do it
            through this argument

            Valid int values:
              0: time.time  |  1: time.perf_counter  |  2: time.proces_time

              if python version >= 3.7:
              3: time.time_ns  |  4: time.perf_counter_ns  |  5: time.proces_time_ns

            Valid str values:
              'time': time.time  |  'perf_counter': time.perf_counter
              'process_time': time.proces_time

              if python version >= 3.7:
              'time_ns': time.time_ns  |  'perf_counter_ns': time.perf_counter_ns
              'proces_time_ns': time.proces_time_ns

            Others:
              Whatever you want to use as time.time
        nested (bool): Allows to do tic toc with nested with a single object.
            If True, you can put several tics using the same object, and each toc will
            correspond to the respective tic.
            If False, it will only register one single tic, and return the respective
            elapsed time of the future tocs.
        print_toc (bool): Indicates if the toc method will print the elapsed time or not.
        """
        self.name = name
        self.nested = nested
        self.tstart = None
        if self.nested:
            self.set_nested(True)

        self._print_toc = print_toc

        self._vsys = sys.version_info

        if self._vsys[0] > 2 and self._vsys[1] >= 7:
            # If python version is greater or equal than 3.7
            self._int2strl = ["time", "perf_counter", "process_time",
                              "time_ns", "perf_counter_ns", "process_time_ns"]
            self._str2fn = {
                "time": [time.time, "s"],
                "perf_counter": [time.perf_counter, "s"],
                "process_time": [time.process_time, "s"],
                "time_ns": [time.time_ns, "ns"],
                "perf_counter_ns": [time.perf_counter_ns, "ns"],
                "process_time_ns": [time.process_time_ns, "ns"],
            }
        elif self._vsys[0] > 2:
            # If python vesion greater than 3
            self._int2strl = ["time", "perf_counter", "process_time"]
            self._str2fn = {
                "time": [time.time, "s"],
                "perf_counter": [time.perf_counter, "s"],
                "process_time": [time.process_time, "s"],
            }
        else:
            # If python version is 2.#
            self._int2strl = ["time"]
            self._str2fn = {"time": [time.time, "s"]}

        if type(method) is not int and type(method) is not str:
            self._get_time = method

        # Parses from integer to string
        if type(method) is int and method < len(self._int2strl):
            method = self._int2strl[method]
        elif type(method) is int and method > len(self._int2strl):
            self._warning_value(method)
            method = "time"

        # Parses from int to the actual timer
        if type(method) is str and method in self._str2fn:
            self._get_time = self._str2fn[method][0]
            self._measure = self._str2fn[method][1]
        elif type(method) is str and method not in self._str2fn:
            self._warning_value(method)
            self._get_time = self._str2fn["time"][0]
            self._measure = self._str2fn["time"][1]

    def __warning_value(self, item):
        msg = "Value '{0}' is not a valid option. Using 'time' instead.".format(
            item)
        warnings.warn(msg, Warning)

    def __enter__(self):
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def __exit__(self, type, value, traceback):
        self.tend = self._get_time()
        if self.nested:
            self.elapsed = self.tend - self.tstart.pop()
        else:
            self.elapsed = self.tend - self.tstart

        self._print_elapsed()

    def _print_elapsed(self):
        """
        Prints the elapsed time
        """
        if self.name != "":
            name = "[{}] ".format(self.name)
        else:
            name = self.name
        printf("-{0}elapsed time: {1:.3g} ({2})".format(name,
               self.elapsed, self._measure))

    def tic(self):
        """
        Defines the start of the timing.
        """
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def toc(self, print_elapsed=None):
        """
        Defines the end of the timing.
        """
        self.tend = self._get_time()
        if self.nested:
            if len(self.tstart) > 0:
                self.elapsed = self.tend - self.tstart.pop()
            else:
                self.elapsed = None
        else:
            if self.tstart:
                self.elapsed = self.tend - self.tstart
            else:
                self.elapsed = None

        if print_elapsed is None:
            if self._print_toc:
                self._print_elapsed()
        else:
            if print_elapsed:
                self._print_elapsed()

        # return(self.elapsed)

    def set_print_toc(self, set_print):
        """
        Indicate if you want the timed time printed out or not.
        Args:
          set_print (bool): If True, a message with the elapsed time will be printed.
        """
        if type(set_print) is bool:
            self._print_toc = set_print
        else:
            warnings.warn(
                "Parameter 'set_print' not boolean. Ignoring the command.", Warning)

    def set_nested(self, nested):
        """
        Sets the nested functionality.
        """
        # Assert that the input is a boolean
        if type(nested) is bool:
            # Check if the request is actually changing the
            # behaviour of the nested tictoc
            if nested != self.nested:
                self.nested = nested

                if self.nested:
                    self.tstart = []
                else:
                    self.tstart = None
        else:
            warnings.warn(
                "Parameter 'nested' not boolean. Ignoring the command.", Warning)


class TicToc2(TicToc):
    def tic(self, nested=True):
        """
        Defines the start of the timing.
        """
        if nested:
            self.set_nested(True)

        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()


__TICTOC_asdfghh123456789 = TicToc2()
tic = __TICTOC_asdfghh123456789.tic
toc = __TICTOC_asdfghh123456789.toc
