from finance_engine.date import constants
import numpy as np

def day():
    return days(1)


def days(n):
    return Period(n, constants.PeriodType.DAY)

def week():
    return weeks(1)


def weeks(n):
    return Period(n, constants.PeriodType.WEEK)


def month():
    return months(1)


def months(n):
    return Period(n, constants.PeriodType.MONTH)


def year():
    return years(1)


def years(n):
    return Period(n, constants.PeriodType.YEAR)


class Period():
    """Represents time periods """
    def __init__(self, quantity, period_type):

        self._quanity = np.array(quantity, np.int32)
        self._period_type = period_type

    def period_type(self):
        return self._period_type
    
    def quantity(self):
        return self._quanity
    
    def __mul__(self, multiplier):
        multiplier = np.array(multiplier, np.int32)
        return Period(self._quanity * multiplier, self._period_type)
    
    def __add__(self, other):
        if other.period_type() != self._period_type:
            raise ValueError("Mixing different period types is not supported")

        return Period(self._quanity + other.quantity(), self._period_type)
    
    def __sub__(self, other):
        if other.period_type() != self._period_type:
            raise ValueError("Mixing different period types is not supported")

        return Period(self._quanity - other.quantity(), self._period_type)

    
    @property
    def shape(self):
        return self._quanity.shape
    
    def _apply_op(self, op_fn):
        q = op_fn(self._quanity)
        return Period(q, self._period_type)
    
    def __repr__(self):
        output = "Period: shape={}".format(self.shape)
        return output

__all__ = [
    "day",
    "days",
    "month",
    "months",
    "week",
    "weeks",
    "year",
    "years",
    "Period",
]