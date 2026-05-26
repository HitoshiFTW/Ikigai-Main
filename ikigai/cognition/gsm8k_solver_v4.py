"""
ikigai.cognition.gsm8k_solver_v4 -- GSM8K Solver V4.

Day 55 Pack 81 -- push V3 (74%) -> 85%+ on hard-50.

V4 adds:
    - Multi-item total cost:    "K items at $P each" -> K * P
    - Buy + change:             "$M, K @ $P each. Change?" -> M - K*P
    - Age subtraction:          "X years old, sister N younger" -> X - N
    - Accumulator:              "$X per week, after N weeks" -> X * N
    - Difference:                "K cats, N dogs. How many more dogs?" -> N - K
    - Distance from speed/time: "X mph for N hours" -> X * N
    - Avg speed:                "T miles in H hours. Average speed?" -> T / H
    - Sales accumulator:        "K per day. After N days?" -> K * N
"""

import re
from ikigai.cognition.gsm8k_solver_v3 import (
    try_fraction, try_percent, try_rate, try_pack_price, try_times_more,
    try_rectangle_area, try_total_cost_with_change, try_round_trip,
    try_per_x_division, try_dozen_subtract, try_weeks_to_days,
    try_time_conversion, try_chain_subtract, try_total_then_subtract,
    _to_num,
)
from ikigai.cognition.verifier import solve_chain, solve_verifier, extract_numbers_smart


# ── new V4 handlers ───────────────────────────────────────────────────────────

def try_multi_item_cost(text):
    """
    "K items at/cost $P each. Total cost?" -> K * P
    Also: "A family of K buys tickets at $P. Total?" -> K * P
    Excludes 'change' (handled separately by try_total_cost_with_change).
    """
    if re.search(r'\bchange\b', text, re.IGNORECASE):
        return None
    if not re.search(r'\b(total|cost|price|paid)\b', text, re.IGNORECASE):
        return None
    if not re.search(r'\bhow much\b|\btotal cost\b', text, re.IGNORECASE):
        return None
    # Find: K items + price each
    # Pattern A: "K items at $P each"
    m = re.search(r'(\d+)\s+\w+\s+at\s+\$?(\d+(?:\.\d+)?)\s+each', text, re.IGNORECASE)
    if m:
        return _to_num(int(m.group(1)) * float(m.group(2)))
    # Pattern B: "ticket costs $P. Family of K"
    cost_m = re.search(r'cost(?:s)?\s+\$?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    fam_m  = re.search(r'family\s+of\s+(\d+)', text, re.IGNORECASE)
    if cost_m and fam_m:
        return _to_num(int(fam_m.group(1)) * float(cost_m.group(1)))
    # Pattern C: "$P per/each X. K X. Total?"
    return None


def try_age_subtract(text):
    """
    "X years old. Sister/brother N years younger. How old is sister?" -> X - N
    """
    if not re.search(r'\byears?\s+old\b', text, re.IGNORECASE):
        return None
    if not re.search(r'\byounger\b', text, re.IGNORECASE):
        return None
    if not re.search(r'\bhow\s+old\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 2:
        return None
    # First number = older age, second = difference
    return _to_num(nums[0] - nums[1])


def try_age_add(text):
    """
    "X years old. Brother N years older. How old is brother?" -> X + N
    """
    if not re.search(r'\byears?\s+old\b', text, re.IGNORECASE):
        return None
    if not re.search(r'\bolder\b', text, re.IGNORECASE):
        return None
    if not re.search(r'\bhow\s+old\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 2:
        return None
    return _to_num(nums[0] + nums[1])


def try_accumulator(text):
    """
    "$X per week. After N weeks. How much saved/total?" -> X * N
    "X per day. After N days?" -> X * N
    """
    period_m = re.search(r'(\d+(?:\.\d+)?)\s+(?:dollars?|\$|\w+)?\s*(?:a|per)\s+(week|day|month|year|hour|minute)',
                         text, re.IGNORECASE)
    if not period_m:
        return None
    base   = float(period_m.group(1))
    period = period_m.group(2).lower()

    after_m = re.search(rf'(?:after|in)\s+(\d+)\s+{period}s?', text, re.IGNORECASE)
    if not after_m:
        return None
    n_periods = int(after_m.group(1))

    if not re.search(r'\b(how much|how many|total)\b', text, re.IGNORECASE):
        return None
    return _to_num(base * n_periods)


def try_difference(text):
    """
    "K X. N Y. How many more Y than X?" -> N - K (positive)
    """
    if not re.search(r'\bhow\s+many\s+more\b', text, re.IGNORECASE):
        return None
    nums = extract_numbers_smart(text)
    if len(nums) < 2:
        return None
    return _to_num(abs(nums[0] - nums[1]))


def try_distance_speed_time(text):
    """
    "X mph for N hours. How far?" -> X * N
    "X miles per hour, 4 hours. How many miles total?" -> X * 4
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:mph|miles\s+per\s+hour|km/h|kph)', text, re.IGNORECASE)
    if not m:
        return None
    speed = float(m.group(1))
    time_m = re.search(r'(\d+(?:\.\d+)?)\s*hours?\b', text, re.IGNORECASE)
    if not time_m:
        return None
    time_h = float(time_m.group(1))
    if not re.search(r'\b(how\s+far|how\s+many\s+miles?|total)\b', text, re.IGNORECASE):
        return None
    return _to_num(speed * time_h)


def try_avg_speed(text):
    """
    "Train leaves 9 AM arrives 1 PM, traveling X miles. Average speed?" -> X / hours
    """
    if not re.search(r'\baverage\s+speed\b', text, re.IGNORECASE):
        return None
    # Time difference from clock
    times = re.findall(r'(\d+)\s*(AM|PM)', text, re.IGNORECASE)
    nums  = extract_numbers_smart(text)
    if len(times) == 2:
        h1, ap1 = int(times[0][0]), times[0][1].upper()
        h2, ap2 = int(times[1][0]), times[1][1].upper()
        # Convert to 24h
        if ap1 == 'PM' and h1 != 12: h1 += 12
        if ap2 == 'PM' and h2 != 12: h2 += 12
        if ap1 == 'AM' and h1 == 12: h1 = 0
        if ap2 == 'AM' and h2 == 12: h2 = 0
        elapsed = h2 - h1
        if elapsed <= 0:
            elapsed += 24
        # Distance = number not equal to the hours
        for n in nums:
            if n > elapsed * 5:   # large enough to be distance
                return _to_num(n / elapsed)
    return None


def try_per_day_total(text):
    """
    "K X per day. After N days, how many X?" -> K * N
    "K X per day, 30-day month" -> K * 30
    Distinct from accumulator: counts items, not money.
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s+\w+\s+per\s+day', text, re.IGNORECASE)
    if not m:
        return None
    per_day = float(m.group(1))
    # Days count
    n_match = re.search(r'(\d+)[\s-]+day', text, re.IGNORECASE)
    after_match = re.search(r'after\s+(\d+)\s+days?', text, re.IGNORECASE)
    days = None
    if after_match:
        days = int(after_match.group(1))
    elif n_match:
        cand = int(n_match.group(1))
        if cand > 1 and abs(cand - per_day) > 1e-6:
            days = cand
    if days is None:
        return None
    if not re.search(r'\bhow\s+many\b|\btotal\b', text, re.IGNORECASE):
        return None
    return _to_num(per_day * days)


# ── V4 main solver ────────────────────────────────────────────────────────────

def solve_v4(text):
    """
    Solver V4: expanded handler set. Returns (answer, method_used).
    """
    handlers = [
        ('fraction',           try_fraction),
        ('percent',            try_percent),
        ('rate',               try_rate),
        ('pack_price',         try_pack_price),
        ('times_more',         try_times_more),
        ('rectangle_area',     try_rectangle_area),
        # V4 new
        ('avg_speed',          try_avg_speed),
        ('distance_st',        try_distance_speed_time),
        ('total_cost_change',  try_total_cost_with_change),
        ('multi_item_cost',    try_multi_item_cost),
        ('age_subtract',       try_age_subtract),
        ('age_add',            try_age_add),
        ('accumulator',        try_accumulator),
        ('per_day_total',      try_per_day_total),
        ('difference',         try_difference),
        ('round_trip',         try_round_trip),
        ('per_x_division',     try_per_x_division),
        ('dozen_subtract',     try_dozen_subtract),
        ('weeks_to_days',      try_weeks_to_days),
        ('time_conversion',    try_time_conversion),
        ('chain_subtract',     try_chain_subtract),
        ('total_then_sub',     try_total_then_subtract),
    ]

    for name, handler in handlers:
        try:
            res = handler(text)
        except Exception:
            res = None
        if res is not None:
            return _to_num(res), name

    # Fallbacks
    try:
        r, _, err = solve_chain(text, max_steps=3)
        if r is not None and err < 1e8:
            return _to_num(r), 'chain'
    except Exception:
        pass

    try:
        v = solve_verifier(text)
        if isinstance(v, tuple) and len(v) >= 1:
            return _to_num(v[0]), 'single'
    except Exception:
        pass

    return None, 'none'
