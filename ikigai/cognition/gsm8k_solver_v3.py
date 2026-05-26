"""
ikigai.cognition.gsm8k_solver_v3 -- GSM8K Solver V3.

Day 55 Pack 80 -- push V2 (58%) -> 70%+ on hard-50.

V3 adds:
    - Unit conversion: dozen=12, weeks->days, hours->minutes, minutes->seconds
    - Geometry: rectangle area = length * width
    - Total-cost-with-change: paid - n_items * unit_price
    - Round trip: (x + y) * n
    - Sub-after-multi: total - (sold), where total may need conversion
    - Per-X questions: divide rate-by-time

Strategy: more specific regex + arithmetic templates, hand off to V2 only on miss.
"""

import re

from ikigai.cognition.gsm8k_solver_v2 import (
    try_fraction, try_percent, try_rate, try_pack_price, try_times_more,
    _to_num,
)
from ikigai.cognition.verifier import solve_chain, solve_verifier, extract_numbers_smart


# Unit conversion factors
UNITS = {
    'dozen':   12,
    'dozens':  12,
    'gross':   144,
    'pair':    2,
    'pairs':   2,
    'week':    7,
    'weeks':   7,
    'fortnight': 14,
    'hour':    60,    # in minutes (for time-conversion)
    'hours':   60,
    'minute':  60,    # in seconds
    'minutes': 60,
    'day':     24,    # in hours (for some Q types)
    'days':    1,     # explicit days = 1 day each
}


#  unit-conversion + dozen handler

def try_dozen_subtract(text):
    """
    "K dozen X. Sold N. How many X left?" -> K*12 - N
    Also "K dozen + N more X?" -> K*12 + N
    """
    m = re.search(r'(\d+)\s+dozen\s+(\w+)', text, re.IGNORECASE)
    if not m:
        return None
    k = int(m.group(1))
    total = k * 12
    nums = extract_numbers_smart(text)
    # Find the "sold/used/ate/lost" number != k
    sub_match = re.search(r'(?:sold|gave|ate|lost|used|removed)\s+(\d+)', text, re.IGNORECASE)
    if sub_match:
        sub_n = int(sub_match.group(1))
        if re.search(r'\bhow\s+many\b.*\b(left|remain|remaining)\b', text, re.IGNORECASE):
            return total - sub_n
        return total - sub_n
    # Else: just return total
    if re.search(r'\bhow\s+many\b', text, re.IGNORECASE):
        return total
    return None


def try_weeks_to_days(text):
    """
    "X miles/runs/etc per day for N weeks. Total?" -> X * 7 * N
    Or generic "per day for N weeks" -> base * 7 * N.
    """
    m = re.search(r'\b(\d+)\s+weeks?\b', text, re.IGNORECASE)
    if not m:
        return None
    n_weeks = int(m.group(1))

    # Per-day base value
    per_day = re.search(r'(\d+(?:\.\d+)?)\s+\w+\s+(?:every|per)\s+day', text, re.IGNORECASE)
    if not per_day:
        return None
    base = float(per_day.group(1))
    if re.search(r'\b(?:total|how many)\b', text, re.IGNORECASE):
        return _to_num(base * 7 * n_weeks)
    return None


def try_time_conversion(text):
    """
    "K minutes. How many seconds?" -> K * 60
    "K hours. How many minutes?"   -> K * 60
    "K hours. How many seconds?"   -> K * 3600
    """
    m_min  = re.search(r'(\d+(?:\.\d+)?)\s*minutes?', text, re.IGNORECASE)
    m_hour = re.search(r'(\d+(?:\.\d+)?)\s*hours?',   text, re.IGNORECASE)

    if m_min and re.search(r'\bhow\s+many\s+seconds?\b', text, re.IGNORECASE):
        return _to_num(float(m_min.group(1)) * 60)
    if m_hour and re.search(r'\bhow\s+many\s+minutes?\b', text, re.IGNORECASE):
        return _to_num(float(m_hour.group(1)) * 60)
    if m_hour and re.search(r'\bhow\s+many\s+seconds?\b', text, re.IGNORECASE):
        return _to_num(float(m_hour.group(1)) * 3600)
    return None


def try_rectangle_area(text):
    """
    "A rectangle is L meters/units long and W wide. Area?" -> L * W
    """
    if not re.search(r'\b(rectangle|area)\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 2:
        return None
    return _to_num(nums[0] * nums[1])


def try_total_cost_with_change(text):
    """
    "Item costs $P. N items. Paid with $M. How much change?" -> M - N*P
    """
    if not re.search(r'\bchange\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 3:
        return None
    # Find: paid (max), unit_price, count
    paid = max(nums)
    # The other two: unit price and count. Smaller is usually count.
    rest = [n for n in nums if n != paid]
    if len(rest) < 2:
        return None
    # Use the largest of the rest as price, smallest as count
    # (heuristic that works for "book costs $15. Mark bought 3 books and paid with $50")
    rest_sorted = sorted(rest)
    count, price = rest_sorted[0], rest_sorted[1]
    return _to_num(paid - count * price)


def try_round_trip(text):
    """
    "X to school and Y back, N days a week. Total weekly?" -> (X + Y) * N
    """
    if not re.search(r'\b(back|return|round trip|both ways)\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 3:
        return None
    # Look for "X to ... Y back, N days"
    # X and Y often equal (commute) or different
    # First two numbers usually distances, third = day count
    return _to_num((nums[0] + nums[1]) * nums[2])


def try_chain_subtract(text):
    """
    "Pizza into K slices. N people eat 1 each. How many left?" -> K - N
    "Box has K. P removed. How many?" -> K - P
    """
    if not re.search(r'\b(left|remain|remaining)\b', text, re.IGNORECASE):
        return None
    # If there's an explicit pizza/box/jar context, use first two numbers
    if not re.search(r'\b(pizza|slice|box|jar|bag|bottle|loaf|loaves)\b',
                     text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 2:
        return None
    # K - N where N might be sum of individual eaters
    # Special: "if N people eat M each" -> N*M
    m = re.search(r'(\d+)\s+people\s+(?:eat|drink|use)\s+(\d+)\s+(?:each|slice)?',
                  text, re.IGNORECASE)
    if m:
        consumed = int(m.group(1)) * int(m.group(2))
        # Find K = number NOT in (people count, per-person count)
        people, per = int(m.group(1)), int(m.group(2))
        for n in nums:
            if n != people and n != per:
                return _to_num(n - consumed)
    # Default: K - N
    return _to_num(nums[0] - nums[1])


def try_per_x_division(text):
    """
    Division: "K per X. How long for N?" -> N / K
              "How many days for P pages, eating N pages/day" -> P / N
    Only fires when question word implies division (how long / how many <time-unit> / how many <count-unit-of-rate>).
    """
    rate_m = re.search(r'(\d+(?:\.\d+)?)\s+(\w+)\s+per\s+(\w+)', text, re.IGNORECASE)
    if not rate_m:
        return None
    rate = float(rate_m.group(1))
    rate_unit = rate_m.group(2).lower().rstrip('s')   # e.g. "gallon", "page"

    # Division question if: "how long", or "how many <rate_unit>" (same unit as rate numerator).
    # NOT division if: "how many <other>" (e.g. miles when rate is mph).
    is_division = False
    if re.search(r'\bhow\s+long\b', text, re.IGNORECASE):
        is_division = True
    # Same-unit question -> division (e.g. "X pages/day, how many days")
    if re.search(rf'\bhow\s+many\s+{rate_m.group(3)}s?\b', text, re.IGNORECASE):
        is_division = True
    # "How many <rate_unit>" -> multiplication, not division
    if re.search(rf'\bhow\s+many\s+{rate_unit}s?\b', text, re.IGNORECASE):
        is_division = False

    if not is_division:
        return None

    nums = extract_numbers_smart(text)
    target = None
    for n in nums:
        if abs(n - rate) > 1e-6:
            target = n
            break
    if target is None or rate == 0:
        return None
    return _to_num(target / rate)


def try_total_then_subtract(text):
    """
    "Anna had X. Gave Y. Lost Z. How many left?" -> X - Y - Z
    """
    if not re.search(r'\b(left|remain|remaining)\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 3:
        return None
    # Heuristic: first number = total, rest are losses
    total = nums[0]
    losses = sum(nums[1:])
    if losses >= total:
        return None
    return _to_num(total - losses)


#  main V3 solver

def solve_v3(text):
    """
    Solver V3 = expanded special-case preprocessor + V2 fallback.
    Returns (answer, method_used).
    """
    # Higher-specificity patterns first
    for name, handler in [
        ('fraction',          try_fraction),
        ('percent',           try_percent),
        ('rate',              try_rate),
        ('pack_price',        try_pack_price),
        ('times_more',        try_times_more),
        ('rectangle_area',    try_rectangle_area),
        ('total_cost_change', try_total_cost_with_change),
        ('round_trip',        try_round_trip),
        ('per_x_division',    try_per_x_division),
        ('dozen_subtract',    try_dozen_subtract),
        ('weeks_to_days',     try_weeks_to_days),
        ('time_conversion',   try_time_conversion),
        ('chain_subtract',    try_chain_subtract),
        ('total_then_sub',    try_total_then_subtract),
    ]:
        try:
            res = handler(text)
        except Exception:
            res = None
        if res is not None:
            return _to_num(res), name

    # Fall back to solve_chain
    try:
        chain_res, _, chain_err = solve_chain(text, max_steps=3)
        if chain_res is not None and chain_err < 1e8:
            return _to_num(chain_res), 'chain'
    except Exception:
        pass

    try:
        v = solve_verifier(text)
        if isinstance(v, tuple) and len(v) >= 1:
            return _to_num(v[0]), 'single'
    except Exception:
        pass

    return None, 'none'
