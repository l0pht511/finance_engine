import numpy as np

from finance_engine.date import constants
from finance_engine.date import tensor_wrapper


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


class Period(tensor_wrapper.TensorWrapper):
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
    
    @classmethod
    def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
        q = op_fn([t.quantity() for t in tensor_wrappers])
        period_type = tensor_wrappers[0].period_type()
        if not all(t.period_type() == period_type for t in tensor_wrappers[1:]):
            raise ValueError("Combined PeriodTensors must have the same PeriodType")
        return Period(q, period_type)
    
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