"""Shared financial calculations and plant-size cost scaling."""

import math


def capital_recovery(rate, years):
    if not all(math.isfinite(v) for v in (rate, years)) or rate < 0 or years <= 0:
        raise ValueError("Discount rate must be non-negative and life positive")
    return 1.0 / years if rate == 0 else rate / (1.0 - (1.0 + rate) ** -years)


def scaled_unit_cost(size, small_size, small_cost, large_size, large_cost):
    """Log-log interpolation between two costed sizes, held flat outside them."""
    if (not all(math.isfinite(v) for v in (size, small_size, small_cost, large_size, large_cost))
            or size < 0 or min(small_size, small_cost, large_cost) <= 0 or large_size <= small_size):
        raise ValueError("Scaled cost needs positive costs and increasing sizes")
    if size <= small_size:
        return small_cost
    if size >= large_size:
        return large_cost
    share = math.log(size / small_size) / math.log(large_size / small_size)
    return small_cost * (large_cost / small_cost) ** share
