# General utility functions for all parts of the pipeline
import time

import requests
import ssl
from urllib3 import poolmanager
from datetime import datetime, timedelta
from datetime import date
from dateutil.rrule import rrule, DAILY, MONTHLY, WEEKLY, YEARLY

DATETIME_FORMAT = "%Y-%m-%d"


def string_to_dt(string):
    """Transforms string to datetime."""
    return datetime.strptime(string, DATETIME_FORMAT)


def dt_to_string(dt):
    """Transforms datetime to string."""
    return datetime.strftime(dt, DATETIME_FORMAT)


valid_forecast_dates = {
    "reforecast": {
        "ecmwf": (string_to_dt("2015-05-14"), datetime.today(), "monday/thursday"),
    },
    "forecast": {
        "ecmwf": (string_to_dt("2015-05-14"), datetime.today(), "monday/thursday"),
    },
}


def is_valid_forecast_date(model, forecast_type, forecast_date):
    assert isinstance(forecast_date, datetime)
    try:
        return forecast_date in generate_dates_in_between(*valid_forecast_dates[forecast_type][model])
    except KeyError:
        return False


def generate_dates_in_between(first_date, last_date, date_frequency):
    if date_frequency == "monday/thursday":
        dates = [
            date
            for date in generate_dates_in_between(first_date, last_date, "daily")
            if date.strftime("%A") in ["Monday", "Thursday"]
        ]
        return dates
    else:
        frequency_to_int = {"daily": 1, "weekly": 7}
        dates = [
            first_date +
            timedelta(days=x * frequency_to_int[date_frequency])
            for x in range(0, int((last_date - first_date).days / (frequency_to_int[date_frequency])) + 1,)
        ]
        return dates


def get_grid(region_id):
    if region_id == "global1_5":
        longitudes = ["0", "358.5"]
        latitudes = ["-90.0", "90.0"]
        grid_size = "1.5"
    elif region_id == "global0_5":
        longitudes = ["0.25", "359.75"]
        latitudes = ["-89.75", "89.75"]
        grid_size = "0.5"
    elif region_id == "us1_0":
        longitudes = ["-125.0", "-67.0"]
        latitudes = ["25.0", "50.0"]
        grid_size = "1.0"
    elif region_id == "us1_5":
        longitudes = ["-123", "-67.5"]
        latitudes = ["25.5", "48"]
        grid_size = "1.5"
    else:
        raise NotImplementedError(
            "Only grids global1_5, us1_0 and us1_5 have been implemented.")
    return longitudes, latitudes, grid_size


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
            r = session.get(url, timeout=timeout, cookies=cookies)
            return r
        except requests.exceptions.Timeout as e:
            # Wait until making another request
            if i == retry - 1:
                raise e
            print(f"Request to url {url} has timed out. Trying again...")
            time.sleep(3)
    print(f"Failed to retrieve file after {retry} attempts. Stopping...")


def get_dates(start_time, end_time, stride="day", return_string=False):
    """Outputs the list of dates corresponding to input date string."""
    # Input is of the form '20170101-20180130'
    start_date = datetime.strptime(start_time, DATETIME_FORMAT)
    end_date = datetime.strptime(end_time, DATETIME_FORMAT)

    if stride == "day":
        stride = DAILY
    elif stride == "week":
        stride = WEEKLY
    elif stride == "month":
        stride = MONTHLY
    elif stride == "year":
        stride = YEARLY
    else:
        raise ValueError(
            "Only day, week, month, and year strides are supported.")
    dates = [dt for dt in rrule(stride, dtstart=start_date, until=end_date)]
    if return_string:
        dates = [date.strftime(DATETIME_FORMAT) for date in dates]
    return dates
