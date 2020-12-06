"""DateFinance definition."""

import collections
import datetime
import numpy as np

from finance_engine.date import constants
from finance_engine.date import date_utils
from finance_engine.date import periods


# Days in each month of a non-leap year.
_DAYS_IN_MONTHS_NON_LEAP = [
    31,  # January.
    28,  # February.
    31,  # March.
    30,  # April.
    31,  # May.
    30,  # June.
    31,  # July.
    31,  # August.
    30,  # September.
    31,  # October.
    30,  # November.
    31,  # December.
]

# Days in each month of a leap year.
_DAYS_IN_MONTHS_LEAP = [
    31,  # January.
    29,  # February.
    31,  # March.
    30,  # April.
    31,  # May.
    30,  # June.
    31,  # July.
    31,  # August.
    30,  # September.
    31,  # October.
    30,  # November.
    31,  # December.
]

# Combined list of days per month. A sentinel value of 0 is added to the top of
# the array so indexing is easier.
_DAYS_IN_MONTHS_COMBINED = [0] + _DAYS_IN_MONTHS_NON_LEAP + _DAYS_IN_MONTHS_LEAP

_ORDINAL_OF_1_1_1970 = 719163


class DateFinance():

    def __init__(self, ordinals, years, months, days):
        self._ordinals = np.array(ordinals, dtype=np.int32)
        self._years = np.array(years, dtype=np.int32)
        self._months = np.array(months, dtype=np.int32)
        self._days = np.array(days, dtype=np.int32)
        self._day_of_year = None  # Computed lazily.
    
    def day(self):
        """Returns an int32 tensor of days since the beginning the month.
        The result is one-based, i.e. yields 1 for first day of the month.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
        dates.day()  # [25, 2]
        ```
        """
        return self._days

    def day_of_week(self):
        """Returns an int32 tensor of weekdays.
        The result is zero-based according to Python datetime convention, i.e.
        Monday is "0".
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
        dates.day_of_week()  # [5, 1]
        ```
        """
        # 1 Jan 0001 was Monday according to the proleptic Gregorian calendar.
        # So, 1 Jan 0001 has ordinal 1, and the weekday is 0.
        return (self._ordinals - 1) % 7

    def month(self):
        """Returns an int32 tensor of months.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
        dates.month()  # [1, 3]
        ```
        """
        return self._months

    def year(self):
        """Returns an int32 tensor of years.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
        dates.year()  # [2019, 2020]
        ```
        """
        return self._years

    def ordinal(self):
        """Returns an int32 tensor of ordinals.
        Ordinal is the number of days since 1st Jan 0001.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2019, 3, 25), (1, 1, 1)])
        dates.ordinal()  # [737143, 1]
        ```
        """
        return self._ordinals
    
    def to_tensor(self):
        """Packs the dates into a single Tensor.
        The Tensor has shape `date_tensor.shape() + (3,)`, where the last dimension
        represents years, months and days, in this order.
        This can be convenient when the dates are the final result of a computation
        in the graph mode: a `tf.function` can return `date_tensor.to_tensor()`, or,
        if one uses `tf.compat.v1.Session`, they can call
        `session.run(date_tensor.to_tensor())`.
        Returns:
        A Tensor of shape `date_tensor.shape() + (3,)`.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
        dates.to_tensor()  # tf.Tensor with contents [[2019, 1, 25], [2020, 3, 2]].
        ```
        """
        return np.stack((self.year(), self.month(), self.day()), axis=-1)
    
    def day_of_year(self):
        """Calculates the number of days since the beginning of the year.
        Returns:
        Tensor of int32 type with elements in range [1, 366]. January 1st yields
        "1".
        #### Example
        ```python
        dt = tff.datetime.dates_from_tuples([(2019, 1, 25), (2020, 3, 2)])
        dt.day_of_year()  # [25, 62]
        ```
        """
        if self._day_of_year is None:
            cumul_days_in_month_nonleap = np.cumsum(
                [0] + _DAYS_IN_MONTHS_NON_LEAP)
            cumul_days_in_month_leap = np.cumsum(
                [0] + _DAYS_IN_MONTHS_LEAP)
            days_before_month_non_leap = np.take(cumul_days_in_month_nonleap,
                                                    self.month() - 1)
            days_before_month_leap = np.take(cumul_days_in_month_leap,
                                                self.month() - 1)
            days_before_month = np.where(
                date_utils.is_leap_year(self.year()), days_before_month_leap,
                days_before_month_non_leap)
            self._day_of_year = days_before_month + self.day()
        return self._day_of_year
    
    def days_until(self, target_date_tensor):
        """Computes the number of days until the target dates.
        Args:
        target_date_tensor: A DateTensor object broadcastable to the shape of
            "self".
        Returns:
        An int32 tensor with numbers of days until the target dates.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2020, 1, 25), (2020, 3, 2)])
        target = tff.datetime.dates_from_tuples([(2020, 3, 5)])
        dates.days_until(target) # [40, 3]
        targets = tff.datetime.dates_from_tuples([(2020, 2, 5), (2020, 3, 5)])
        dates.days_until(targets)  # [11, 3]
        ```
        """
        return target_date_tensor.ordinal() - self._ordinals
    
    def period_length_in_days(self, period):
        """Computes the number of days in each period.
        Args:
        period_tensor: A PeriodTensor object broadcastable to the shape of "self".
        Returns:
        An int32 tensor with numbers of days each period takes.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2020, 2, 25), (2020, 3, 2)])
        dates.period_length_in_days(month())  # [29, 31]
        periods = tff.datetime.months([1, 2])
        dates.period_length_in_days(periods)  # [29, 61]
        ```
        """
        return (self + period).ordinal() - self._ordinals
    
    def is_end_of_month(self):
        """Returns a bool Tensor indicating whether dates are at ends of months."""
        return np.equal(self._days,
                            _num_days_in_month(self._months, self._years))
    
    def to_end_of_month(self):
        """Returns a new DateTensor with each date shifted to the end of month."""
        days = _num_days_in_month(self._months, self._years)
        return from_year_month_day(self._years, self._months, days, validate=False)
    
    @property
    def shape(self):
        return self._ordinals.shape
    
    def __add__(self, period_tensor):
        """Adds a tensor of periods.
        When adding months or years, the resulting day of the month is decreased
        to the largest valid value if necessary. E.g. 31.03.2020 + 1 month =
        30.04.2020, 29.02.2020 + 1 year = 28.02.2021.
        Args:
        period_tensor: A `PeriodTensor` object broadcastable to the shape of
        "self".
        Returns:
        The new instance of DateTensor.
        #### Example
        ```python
        dates = tff.datetime.dates_from_tuples([(2020, 2, 25), (2020, 3, 31)])
        new_dates = dates + tff.datetime.month()
        # DateTensor([(2020, 3, 25), (2020, 4, 30)])
        new_dates = dates + tff.datetime.month([1, 2])
        # DateTensor([(2020, 3, 25), (2020, 5, 31)])
        ```
        """
        period_type = period_tensor.period_type()

        if period_type == constants.PeriodType.DAY:
            ordinals = self._ordinals + period_tensor.quantity()
            return from_ordinals(ordinals)

        if period_type == constants.PeriodType.WEEK:
            return self + periods.Period(period_tensor.quantity() * 7,
                                                constants.PeriodType.DAY)

        def adjust_day(year, month, day):
            return np.minimum(day, _num_days_in_month(month, year))

        if period_type == constants.PeriodType.MONTH:
            m = self._months - 1 + period_tensor.quantity()
            y = self._years + m // 12
            m = m % 12 + 1
            d = adjust_day(y, m, self._days)
            return from_year_month_day(y, m, d, validate=False)

        if period_type == constants.PeriodType.YEAR:
            y = self._years + period_tensor.quantity()
            # Use tf.shape to handle the case of dynamically shaped `y`
            m = np.broadcast_to(self._months, np.shape(y))
            d = adjust_day(y, m, self._days)
            return from_year_month_day(y, m, d, validate=False)

        raise ValueError("Unrecognized period type: {}".format(period_type))

    def __sub__(self, period_tensor):
        """Subtracts a tensor of periods.
        When subtracting months or years, the resulting day of the month is
        decreased to the largest valid value if necessary. E.g. 31.03.2020 - 1 month
        = 29.02.2020, 29.02.2020 - 1 year = 28.02.2019.
        Args:
        period_tensor: a PeriodTensor object broadcastable to the shape of "self".
        Returns:
        The new instance of DateTensor.
        """
        return self + periods.Period(-period_tensor.quantity(),
                                        period_tensor.period_type())
    
    def __eq__(self, other):
        """Compares two DateTensors by "==", returning a Tensor of bools."""
        # Note that tf doesn't override "==" and  "!=", unlike numpy.
        return np.equal(self._ordinals, other.ordinal())

    def __ne__(self, other):
        """Compares two DateTensors by "!=", returning a Tensor of bools."""
        return np.not_equal(self._ordinals, other.ordinal())

    def __gt__(self, other):
        """Compares two DateTensors by ">", returning a Tensor of bools."""
        return self._ordinals > other.ordinal()

    def __ge__(self, other):
        """Compares two DateTensors by ">=", returning a Tensor of bools."""
        return self._ordinals >= other.ordinal()

    def __lt__(self, other):
        """Compares two DateTensors by "<", returning a Tensor of bools."""

        return self._ordinals < other.ordinal()

    def __le__(self, other):
        """Compares two DateTensors by "<=", returning a Tensor of bools."""
        return self._ordinals <= other.ordinal()

    def __repr__(self):
        output = "DateTensor: shape={}".format(self.shape)
        contents_np = np.stack((self._years, self._months, self._days), axis=-1)
        return output + ", contents={}".format(repr(contents_np))


def _num_days_in_month(month, year):
    """Returns number of days in a given month of a given year."""
    days_in_months = np.array(_DAYS_IN_MONTHS_COMBINED, np.int32)
    is_leap = date_utils.is_leap_year(year)
    return np.take(days_in_months,
                     month + 12 * is_leap.astype(np.int32))


def convert_to_date_tensor(date_inputs):
    """Converts supplied data to a `DateTensor` if possible.
    Args:
        date_inputs: One of the supported types that can be converted to a
        DateTensor. The following input formats are supported. 1. Sequence of
        `datetime.datetime`, `datetime.date`, or any other structure with data
        attributes called 'year', 'month' and 'day'. 2. A numpy array of
        `datetime64` type. 3. Sequence of (year, month, day) Tuples. Months are
        1-based (with January as 1) and constants.Months enum may be used instead
        of ints. Days are also 1-based. 4. A tuple of three int32 `Tensor`s
        containing year, month and date as positive integers in that order. 5. A
        single int32 `Tensor` containing ordinals (i.e. number of days since 31
        Dec 0 with 1 being 1 Jan 1.)
    Returns:
        A `DateTensor` object representing the supplied dates.
    Raises:
        ValueError: If conversion fails for any reason.
    """
    if isinstance(date_inputs, DateFinance):
        return date_inputs

    if hasattr(date_inputs, "year"):  # Case 1.
        return from_datetimes(date_inputs)

    if isinstance(date_inputs, np.ndarray):  # Case 2.
        date_inputs = date_inputs.astype("datetime64[D]")
        return from_np_datetimes(date_inputs)

    if isinstance(date_inputs, collections.Sequence):
        if not date_inputs:
            return from_ordinals([])
        test_element = date_inputs[0]
        if hasattr(test_element, "year"):  # Case 1.
            return from_datetimes(date_inputs)
        # Case 3
        if isinstance(test_element, collections.Sequence):
            return from_tuples(date_inputs)
        if len(date_inputs) == 3:  # Case 4.
            return from_year_month_day(date_inputs[0], date_inputs[1], date_inputs[2])
    # As a last ditch effort, try to convert the sequence to a Tensor to see if
    # that can work
    try:
        as_ordinals = np.array(date_inputs, dtype=np.int32)
        return from_ordinals(as_ordinals)
    except ValueError as e:
        raise ValueError("Failed to convert inputs to DateTensor. "
                        "Unrecognized format. Error: " + e)

def from_datetimes(datetimes):
    """Creates DateTensor from a sequence of Python datetime objects.
    Args:
        datetimes: Sequence of Python datetime objects.
    Returns:
        DateTensor object.
    #### Example
    ```python
    import datetime
    dates = [datetime.date(2015, 4, 15), datetime.date(2017, 12, 30)]
    date_tensor = tff.datetime.dates_from_datetimes(dates)
    ```
    """
    if isinstance(datetimes, (datetime.date, datetime.datetime)):
        return from_year_month_day(datetimes.year, datetimes.month, datetimes.day,
                                validate=False)
    years = np.array([dt.year for dt in datetimes], dtype=np.int32)
    months = np.array([dt.month for dt in datetimes], dtype=np.int32)
    days = np.array([dt.day for dt in datetimes], dtype=np.int32)

    # datetime stores year, month and day internally, and datetime.toordinal()
    # performs calculations. We use a tf routine to perform these calculations
    # instead.
    return from_year_month_day(years, months, days, validate=False)

def from_np_datetimes(np_datetimes):
    """Creates DateTensor from a Numpy array of dtype datetime64.
    Args:
        np_datetimes: Numpy array of dtype datetime64.
    Returns:
        DateTensor object.
    #### Example
    ```python
    import datetime
    import numpy as np
    date_tensor_np = np.array(
        [[datetime.date(2019, 3, 25), datetime.date(2020, 6, 2)],
        [datetime.date(2020, 9, 15), datetime.date(2020, 12, 27)]],
        dtype=np.datetime64)
    date_tensor = tff.datetime.dates_from_np_datetimes(date_tensor_np)
    ```
    """
    # There's no easy way to extract year, month, day from numpy datetime, so
    # we start with ordinals.
    ordinals = np.array(np_datetimes, dtype=np.int32) + _ORDINAL_OF_1_1_1970
    return from_ordinals(ordinals, validate=False)


def from_tuples(year_month_day_tuples, validate=True):
    """Creates DateTensor from a sequence of year-month-day Tuples.
    Args:
        year_month_day_tuples: Sequence of (year, month, day) Tuples. Months are
        1-based; constants from Months enum can be used instead of ints. Days are
        also 1-based.
        validate: Whether to validate the dates.
    Returns:
        DateTensor object.
    #### Example
    ```python
    date_tensor = tff.datetime.dates_from_tuples([(2015, 4, 15), (2017, 12, 30)])
    ```
    """
    years, months, days = [], [], []
    for t in year_month_day_tuples:
        years.append(t[0])
        months.append(t[1])
        days.append(t[2])
    years = np.array(years, dtype=np.int32)
    months = np.array(months, dtype=np.int32)
    days = np.array(days, dtype=np.int32)
    return from_year_month_day(years, months, days, validate)





def from_year_month_day(year, month, day, validate=True):
    """Creates DateTensor from tensors of years, months and days.
    Args:
        year: Tensor of int32 type. Elements should be positive.
        month: Tensor of int32 type of same shape as `year`. Elements should be in
        range `[1, 12]`.
        day: Tensor of int32 type of same shape as `year`. Elements should be in
        range `[1, 31]` and represent valid dates together with corresponding
        elements of `month` and `year` Tensors.
        validate: Whether to validate the dates.
    Returns:
        DateTensor object.
    #### Example
    ```python
    year = tf.constant([2015, 2017], dtype=tf.int32)
    month = tf.constant([4, 12], dtype=tf.int32)
    day = tf.constant([15, 30], dtype=tf.int32)
    date_tensor = tff.datetime.dates_from_year_month_day(year, month, day)
    ```
    """
    year = np.array(year, np.int32)
    month = np.array(month, np.int32)
    day = np.array(day, np.int32)

    # control_deps = []
    # if validate:
    #     control_deps.append(
    #         tf.debugging.assert_positive(year, message="Year must be positive."))
    #     control_deps.append(
    #         tf.debugging.assert_greater_equal(
    #             month,
    #             constants.Month.JANUARY.value,
    #             message=f"Month must be >= {constants.Month.JANUARY.value}"))
    #     control_deps.append(
    #         tf.debugging.assert_less_equal(
    #             month,
    #             constants.Month.DECEMBER.value,
    #             message="Month must be <= {constants.Month.JANUARY.value}"))
    #     control_deps.append(
    #         tf.debugging.assert_positive(day, message="Day must be positive."))
    #     is_leap = date_utils.is_leap_year(year)
    #     days_in_months = tf.constant(_DAYS_IN_MONTHS_COMBINED, tf.int32)
    #     max_days = tf.gather(days_in_months,
    #                         month + 12 * tf.dtypes.cast(is_leap, np.int32))
    #     control_deps.append(
    #         tf.debugging.assert_less_equal(
    #             day, max_days, message="Invalid day-month pairing."))
    #     with tf.compat.v1.control_dependencies(control_deps):
    #         # Ensure years, months, days themselves are under control_deps.
    #         year = tf.identity(year)
    #         month = tf.identity(month)
    #         day = tf.identity(day)

    ordinal = date_utils.year_month_day_to_ordinal(year, month, day)
    return DateFinance(ordinal, year, month, day)


def from_ordinals(ordinals, validate=True):
    """Creates DateTensor from tensors of ordinals.
    Args:
        ordinals: Tensor of type int32. Each value is number of days since 1 Jan
        0001. 1 Jan 0001 has `ordinal=1`.
        validate: Whether to validate the dates.
    Returns:
        DateTensor object.
    #### Example
    ```python
    ordinals = tf.constant([
        735703,  # 2015-4-12
        736693   # 2017-12-30
    ], dtype=tf.int32)
    date_tensor = tff.datetime.dates_from_ordinals(ordinals)
    ```
    """
    ordinals = np.array(ordinals, dtype=np.int32)

    # control_deps = []
    # if validate:
    #     control_deps.append(
    #         tf.debugging.assert_positive(
    #             ordinals, message="Ordinals must be positive."))
    #     with tf.compat.v1.control_dependencies(control_deps):
    #     ordinals = tf.identity(ordinals)

    years, months, days = date_utils.ordinal_to_year_month_day(ordinals)
    return DateFinance(ordinals, years, months, days)