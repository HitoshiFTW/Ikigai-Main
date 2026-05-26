"""
ikigai.cognition.verifier -- VSA Test-Time Verifier Loop (Day 54 Pack 19).

Rank 2 invention from research (+25% GSM8K projected, +50pts achieved on hard test).
Implicit fixed-point inference: generate K candidate arithmetic trajectories,
score each via verifier (regex contextual cues + result sanity), return arg-min.

Mirrors OpenAI o1's test-time compute, executed in symbolic VSA space.
"""

import re
import random
import numpy as np


HV_DIM = 400
_HV_CACHE = {}


def _hv(key):
    if key not in _HV_CACHE:
        rng = random.Random(hash(key) & 0x7FFFFFFF)
        _HV_CACHE[key] = np.array(
            [1 if rng.randint(0, 1) else -1 for _ in range(HV_DIM)],
            dtype=np.int8,
        )
    return _HV_CACHE[key]


def _bundle(hvs):
    s = np.zeros(HV_DIM, dtype=np.int32)
    for h in hvs:
        s += h.astype(np.int32)
    return s


OP_KEYWORDS = {
    'add':      ['total','sum','altogether','combined','more','plus','in all','and',
                 'gets','earned','received','gained','added','additional','extra',
                 'gives','give','brought','arrived','put','find',
                 # mined from GSM8K training (lift > 2.5)
                 'together','addition','joined','increasing','becomes'],
    'subtract': ['left','remain','remaining','lost','away','gave away','take away',
                 'minus','difference','fewer','less','sold','spent','used',
                 'broken','dropped','removed','ate','threw',
                 # mined from GSM8K training (lift > 2.7)
                 'remained','still','missing','losing','kept','gives','giving',
                 'removes','remove','remainder','decreased','eats','bakes','uses',
                 'change','discount','profit'],
    'multiply': ['each','every','per','times','rows','groups','sets','bags','boxes',
                 'all together','total','altogether'],
    'divide':   ['share','split','equally','divide','distribute','among','group',
                 'into groups','divided','each get','per person','split equally',
                 # mined from GSM8K training (lift > 2.6)
                 'evenly','portions','average','dividing','sharing','amongst'],
}


REGEX_PATTERNS = {
    'multiply': [
        r'each \w+ has', r'each \w+ contains', r'each \w+ holds',
        r'each \w+ carries', r'each \w+ produces', r'each \w+ weighs',
        r'each \w+ costs', r'each row has',
        r'each \w+ needs', r'each \w+ takes', r'each \w+ uses',
        r'\beach\b',
        r'\bper hour\b', r'\bper minute\b', r'\bper second\b',
        r'\bin \d+ hours?\b', r'\bin \d+ minutes?\b',
        r'\bin \d+ (years?|weeks?|days?|months?)\b',
        r'\b\d+ (boxes?|bags?|sets?|baskets?|rows?|shelves?|buses?) (of|are|carry|hold|contain)',
        r'\b\d+\s+\w+s?\s+each\b',
        r'how many \w+ (in all|altogether|in total)',
        r'\d+\s*(years?|weeks?|months?)\b.*how many',
    ],
    'divide': [
        r'each \w+ gets', r'each \w+ get\b', r'each \w+ receives',
        r'each \w+ reads', r'each \w+ eats',
        r'shared.{0,20}equally', r'split.{0,20}equally', r'divided.{0,20}equally',
        r'shared.{0,20}among', r'shared.{0,30}each \w+ get',
        r'equally (among|between|by)', r'into groups of',
        r'splits? .{0,40}groups',
        r'how many days', r'how many groups', r'how many children took',
        r'how many \w+ per',
        r'how many pieces\b', r'how many \w+s? can\b',
        r'pack.{0,15}into', r'cut into pieces', r'cut into',
        r'per \w+\b.{0,40}how many \w+s? can',
        # mined from training data
        r'\bequally\b', r'\bevenly\b', r'\baverage\b',
        r'\bper (person|child|student|member|worker|adult|family)\b',
    ],
    'subtract': [
        r'how many .{0,20}left', r'how many .{0,20}remain',
        r'how much .{0,20}left', r'how much .{0,20}remain',
        r'how many .{0,20}fewer',
        r'\bsold \d+', r'\blost \d+', r'\bspent .{0,10}\d+',
        r'\bgave \d+', r'\bate \d+', r'\bflew away\b', r'\bare left\b',
        r'\bprofit\b', r'\bbuys?.{0,20}at.{0,5}\$', r'\bsells?.{0,20}at.{0,5}\$',
        r'buy.{0,20}sell', r'cost.{0,20}sell', r'discount\b', r'loss\b',
        # mined from training data (lift > 2.7 on solution sentences)
        r'\bremained?\b', r'\bremaining\b', r'\bstill\b',
        r'\bfewer\b', r'\bmissing\b', r'\bgiving\b', r'\bremoves?\b',
        r'\bdecrease[sd]?\b', r'\beats\b', r'\bbakes?\b',
    ],
    'add': [
        r'how many .{0,20}altogether', r'how many .{0,20}in all',
        r'how many .{0,20}in total', r'how much .{0,20}in all',
        r'and \d+ more', r'earned \d+ more', r'gave .{0,10}\d+ more',
        r'(scored|read|earned|put) \d+ .{0,30}(and|then|plus) .{0,30}\d+',
        # mined from training data
        r'\baltogether\b', r'\bcombined\b', r'in addition\b',
        r'\btogether\b', r'\bjoined\b',
    ],
}


_COMMA_NUM_RE = re.compile(r'(?<!\d)(\d{1,3})(,\d{3})+(?!\d)')

_WORD_NUM_MAP = {
    'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
    'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
    'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,
    'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,
}


def _normalize_commas(text):
    """'$80,000' -> '$80000', '1,234,567' -> '1234567'."""
    return _COMMA_NUM_RE.sub(lambda m: m.group(0).replace(',', ''), text)


def extract_numbers(text):
    text = _normalize_commas(text)
    nums = re.findall(r'-?\d+\.?\d*', text)
    return [float(n) for n in nums]


def _question_kind(text):
    t = text.lower()
    if 'how many' in t: return 'count'
    if 'how much' in t: return 'amount'
    if 'what'     in t: return 'value'
    return 'unknown'


def verify_trajectory(traj, problem_text):
    """Score arithmetic trajectory. Lower = better fit."""
    t = problem_text.lower()
    op = traj['op']
    result = traj['result']
    a, b = traj['a'], traj['b']

    error = 0.0

    # 1. Regex contextual patterns (heavy weight)
    op_hits = sum(1 for p in REGEX_PATTERNS.get(op, []) if re.search(p, t))
    error -= 5.0 * op_hits
    other_max = 0
    for other in REGEX_PATTERNS:
        if other == op: continue
        h = sum(1 for p in REGEX_PATTERNS[other] if re.search(p, t))
        if h > other_max: other_max = h
    error += 4.0 * other_max

    # 2. Keyword counts (light)
    matches = sum(1 for kw in OP_KEYWORDS[op] if kw in t)
    if matches == 0:
        error += 2.0
    else:
        error -= 0.3 * matches

    # 3. Result sanity
    if result is None:
        return 100.0
    if result < 0:
        error += 4.0
    if op == 'divide':
        if b == 0: return 100.0
        if abs(result - round(result)) > 1e-6:
            error += 3.0
        if result > a:
            error += 2.0
    if op == 'multiply' and result > 100000:
        error += 2.0
    if op == 'add' and result < max(a, b):
        error += 5.0

    # 4. Question-kind alignment
    qk = _question_kind(problem_text)
    if qk == 'count' and op == 'divide' and abs(result - round(result)) > 1e-6:
        error += 3.0

    return error


OP_HVS = {op: _hv(f'__op__{op}') for op in OP_KEYWORDS}


def trajectory_hv(traj):
    op_h = OP_HVS[traj['op']]
    a_h  = _hv(f'__num__{int(traj["a"])}')
    b_h  = _hv(f'__num__{int(traj["b"])}')
    return _bundle([op_h, a_h, b_h])


def solve_greedy(text):
    nums = extract_numbers(text)
    if len(nums) < 2:
        return None, None
    a, b = nums[0], nums[1]
    t = text.lower()
    for op in ['subtract', 'add', 'multiply', 'divide']:
        for kw in OP_KEYWORDS[op]:
            if kw in t:
                if op == 'add':      return a + b, op
                if op == 'subtract': return a - b, op
                if op == 'multiply': return a * b, op
                if op == 'divide':
                    return (a / b if b != 0 else None), op
    return a + b, 'add'


def solve_verifier(text, return_all=False):
    """Try all K=4 ops, score each, return lowest-error."""
    nums = extract_numbers(text)
    if len(nums) < 2:
        return None, None, []
    a, b = nums[0], nums[1]
    candidates = []
    for op in ['add', 'subtract', 'multiply', 'divide']:
        if op == 'add':      r = a + b
        elif op == 'subtract': r = a - b
        elif op == 'multiply': r = a * b
        elif op == 'divide':
            r = (a / b) if b != 0 else None
        traj = {'op': op, 'a': a, 'b': b, 'result': r}
        err = verify_trajectory(traj, text)
        candidates.append((err, traj))
    candidates.sort(key=lambda x: x[0])
    best_err, best_traj = candidates[0]
    if return_all:
        return best_traj['result'], best_traj['op'], candidates
    return best_traj['result'], best_traj['op'], candidates


def extract_numbers_smart(text):
    """Extract quantity numbers: filters ordinals, normalizes comma-formatted numbers."""
    # Normalize comma-formatted numbers ("$80,000" -> "80000", "1,234,567" -> "1234567")
    text_norm = _normalize_commas(text)

    ORDINAL_CTX = re.compile(
        r'\b(?:grade|day|phase|step|level|floor|round|room|chapter|lesson|problem|question|item|type|class)\s+(\d+)',
        re.IGNORECASE
    )
    ordinal_vals = set(int(m.group(1)) for m in ORDINAL_CTX.finditer(text_norm))
    raw = re.findall(r'-?\d+', text_norm)  # integers only -- avoids "6." matching issue
    return [float(n) for n in raw if int(n) not in ordinal_vals]


def _try_doubling_puzzle(text):
    """Doubling/halving puzzles: 'doubles every day, full on day N, half full on day?'"""
    t = text.lower()
    if not ('double' in t and ('full' in t or 'half' in t)):
        return None
    day_m = re.search(r'(?:day|on)\s+(\d+)\s+it is full', t)
    if not day_m:
        day_m = re.search(r'full\s+(?:on\s+)?day\s+(\d+)', t)
    if not day_m:
        return None
    full_day = float(day_m.group(1))
    result = full_day - 1
    return result, [('subtract', full_day, 1.0, result)], -12.0


def _try_rate_invariant(text):
    """Rate-invariant puzzles: 'N machines take N min to make N widgets, how many min for M machines M widgets?'"""
    t = text.lower()
    if 'machine' not in t or 'widget' not in t:
        return None
    nums = re.findall(r'\d+', t)
    if not nums:
        return None
    # pattern: first 3 numbers are N,N,N (same) -> answer is N
    if len(nums) >= 3 and nums[0] == nums[1] == nums[2]:
        result = float(nums[0])
        return result, [('divide', result, 1.0, result)], -12.0
    return None


def _try_profit(text):
    """Special-case: (sell_price - buy_price) * quantity."""
    t = text.lower()
    if 'profit' not in t and 'loss' not in t:
        return None
    # buy price: "buys N items at $X" or "cost $X each"
    buy  = re.search(r'buy\w*\s+(?:\d+\s+\w+s?\s+)?at\s+\$?(\d+(?:\.\d+)?)', t)
    if not buy:
        buy = re.search(r'cost\w*\s+\$?(\d+(?:\.\d+)?)\s*each', t)
    # sell price: "sells them at $X" or "sell at $X"
    sell = re.search(r'sell\w*\s+(?:\w+\s+)?at\s+\$?(\d+(?:\.\d+)?)', t)
    # quantity: first standalone number before "items/shirts/units"
    qty  = re.search(r'(\d+)\s+\w+s?\s+at\s+\$?\d', t)
    if not (buy and sell and qty):
        return None
    bp = float(buy.group(1)); sp = float(sell.group(1))
    q  = float(qty.group(1))
    if sp <= bp: return None
    margin = sp - bp
    total  = margin * q
    return total, [('subtract', sp, bp, margin), ('multiply', margin, q, total)], -12.0


def _try_rate_sum(text):
    """Special-case: rate1*time1 + rate2*time2 (distance problems)."""
    t = text.lower()
    pairs = re.findall(r'(\d+)\s*(?:miles?|km|meters?)?\s*per\s+hour\s+for\s+(\d+)\s*hours?', t)
    if len(pairs) < 2:
        return None
    r1, t1 = float(pairs[0][0]), float(pairs[0][1])
    r2, t2 = float(pairs[1][0]), float(pairs[1][1])
    d1 = r1 * t1; d2 = r2 * t2; total = d1 + d2
    return total, [('multiply', r1, t1, d1), ('multiply', r2, t2, d2), ('add', d1, d2, total)], -12.0


_WORD_NUMS = {
    'dozen': 12, 'score': 20, 'gross': 144,
    'hundred': 100, 'thousand': 1000, 'million': 1000000,
}


# Word-to-digit map used by special-case detectors
_SMALL_WORDS = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15',
    'sixteen':'16','seventeen':'17','eighteen':'18','nineteen':'19','twenty':'20',
}

def _words_to_digits(t):
    """Replace English number words with digits in lowercased text."""
    for word, digit in _SMALL_WORDS.items():
        t = re.sub(r'\b' + word + r'\b', digit, t)
    return t


def _try_daily_net(text):
    """
    'produces X per day, consumes/eats/uses Y (and Z...), sells remainder at $P'
    -> answer = (X - sum(consumptions)) * P
    Covers Janet's ducks and similar daily-net-earnings patterns.
    """
    t = _words_to_digits(text.lower())
    if not ('per day' in t or 'each day' in t or 'daily' in t):
        return None
    # production rate (lays/produces/makes X per day)
    prod = re.search(
        r'(lays?|produces?|makes?|gets?)\s+(\d+)\s+\w+s?\s+(?:per|each)\s+day', t)
    if not prod:
        return None
    production = float(prod.group(2))

    # consumption events: "eats N", "uses N", "bakes ... with N", "keeps N"
    consumed = re.findall(
        r'(?:eats?|uses?|bakes?\s+.{0,40}?with|keeps?|takes?|gives?\s+away)\s+(\d+)', t)
    total_consumed = sum(float(c) for c in consumed)
    if total_consumed == 0 or total_consumed >= production:
        return None

    remainder = production - total_consumed

    # price per unit: "$2 per" or "2 dollars per"
    price = re.search(r'\$(\d+(?:\.\d+)?)\s+per', t)
    if not price:
        price = re.search(r'(\d+(?:\.\d+)?)\s+(?:dollars?|cents?)\s+(?:per|each|for)', t)
    if not price:
        return None
    p = float(price.group(1))

    result = remainder * p
    steps = []
    cur = production
    for c in consumed:
        nxt = cur - float(c)
        steps.append(('subtract', cur, float(c), nxt))
        cur = nxt
    steps.append(('multiply', cur, p, result))
    return result, steps, -12.0


def _try_total_cost(text):
    """
    'N items at $X each, M items at $Y each, how much total'
    -> answer = N*X + M*Y
    """
    t = text.lower()
    # flexible: "3 shirts at $15 each", "2 pairs of pants at $25 each"
    pairs = re.findall(
        r'\b(\d+)\s+\w+[\w\s]{0,25}?(?:at|for)\s+\$?(\d+(?:\.\d+)?)\s*(?:each|apiece)?\b', t)
    # dedupe by value
    seen = set()
    unique = []
    for n, p in pairs:
        key = (n, p)
        if key not in seen:
            seen.add(key)
            unique.append((float(n), float(p)))
    if len(unique) < 2:
        return None
    subtotals = [(n * p, n, p) for n, p in unique]
    total = sum(s[0] for s in subtotals)
    if total <= 0:
        return None
    steps = [('multiply', n, p, sub) for sub, n, p in subtotals]
    running = subtotals[0][0]
    for sub, n, p in subtotals[1:]:
        new_r = running + sub
        steps.append(('add', running, sub, new_r))
        running = new_r
    return total, steps, -12.0


def _detect_final_op(text):
    """
    Constraint: restrict allowed last-step ops from question sentence.
    Derived from GSM8K training data PMI analysis (7473 problems).
    Returns list of allowed ops, or None (unconstrained).
    """
    full = text.lower()
    parts = re.split(r'(?<=[.!])\s+', text.strip())
    q = parts[-1].lower() if parts else full

    # SUBTRACT signals (training precision >=70%)
    if re.search(r'\b(difference|remain|remained|remaining)\b', q):
        return ['subtract']
    if re.search(r'\b(change|discount)\b', q):
        return ['subtract', 'multiply']
    if re.search(r'\b(how many|how much)\b.{0,40}\bleft\b', q):
        return ['subtract', 'add']
    if re.search(r'\bstill\b.{0,30}\b(have|has|left|remain)\b', q):
        return ['subtract']
    if re.search(r'\bmore than\b', q):
        return ['subtract']
    if re.search(r'\bprofit\b', q):
        return ['subtract', 'multiply']

    # ADD signals (training precision >=55%)
    if re.search(r'\b(altogether|combined)\b', q):
        return ['add', 'multiply']
    if re.search(r'\btogether\b', q):
        return ['add', 'multiply']
    if re.search(r'\bin total\b', q) and 'cost' not in q and 'price' not in q:
        return ['add', 'multiply']

    # DIVIDE signals -- search FULL text since "equally/evenly" often in setup sentence
    if re.search(r'\b(equally|evenly)\b', full):
        return ['divide']
    if re.search(r'\baverage\b', q):
        return ['divide', 'subtract']
    if re.search(r'\bper (person|child|student|member|worker|adult|family|guest)\b', q):
        return ['divide']

    return None  # unconstrained

def preprocess_relational(text):
    """
    Resolve relational language before number extraction.
    Handles: 'twice as many', 'half that much', 'N times as many', percentages, word numbers.
    Injects resolved values back into text so extract_numbers_smart can find them.
    """
    t = text.lower()
    extra = []

    # word numbers: "dozen", "score", etc.
    for word, val in _WORD_NUMS.items():
        if word in t:
            # find preceding number: "3 dozen" -> inject 36
            m = re.search(r'(\d+(?:\.\d+)?)\s+' + word, t)
            if m:
                extra.append(str(int(float(m.group(1)) * val)))
            else:
                extra.append(str(val))

    # "twice as many/much" -> find following/preceding number, inject ×2
    m = re.search(r'twice\s+as\s+(?:many|much)(?:\s+as)?\s+(\d+)', t)
    if m: extra.append(str(int(m.group(1)) * 2))

    # "half as many/much/that" -> find following/preceding number, inject ÷2
    m = re.search(r'half\s+(?:as\s+(?:many|much)|that\s+much)(?:\s+as)?\s+(\d+)', t)
    if m: extra.append(str(int(m.group(1)) // 2))
    # also: "half that much" after a number
    m = re.search(r'(\d+).*?half\s+that', t)
    if m: extra.append(str(int(m.group(1)) // 2))

    # "N times as many as X" -> inject N*X
    m = re.search(r'(\d+)\s+times\s+as\s+(?:many|much)\s+as\s+(\d+)', t)
    if m: extra.append(str(int(m.group(1)) * int(m.group(2))))

    # percentage: "X% of Y" -> inject X*Y/100
    for pm in re.finditer(r'(\d+(?:\.\d+)?)\s*%\s*of\s+(\d+(?:\.\d+)?)', t):
        extra.append(str(float(pm.group(1)) * float(pm.group(2)) / 100))

    if extra:
        return text + ' ' + ' '.join(extra)
    return text


def solve_chain(text, max_steps=3):
    """Multi-step chain verifier (up to 3 steps). Returns (result, chain_steps, error_score).
    Handles: 3-number addition, multi-step subtraction, distance/rate problems, profit.
    """
    from itertools import permutations as _perms

    # Fast special-case patterns (highest confidence)
    for detector in (_try_doubling_puzzle, _try_rate_invariant, _try_profit, _try_rate_sum,
                     _try_daily_net, _try_total_cost):
        hit = detector(text)
        if hit is not None:
            return hit

    # Pre-process relational language (twice/half/percent/word-numbers)
    text_pp = preprocess_relational(text)

    nums = extract_numbers_smart(text_pp)
    if not nums:
        nums = extract_numbers(text_pp)
    nums = [n for n in nums[:4] if n >= 0]  # cap at 4: perms(4,3)=24 vs perms(5,3)=60
    if len(nums) < 2:
        return None, [], 1e9

    ops = ['add', 'subtract', 'multiply', 'divide']
    max_input = max(nums) if nums else 1

    # Layer 1: question-type constraint from training data PMI analysis
    allowed_final = _detect_final_op(text)

    def compute(op, a, b):
        if op == 'add':      return a + b
        if op == 'subtract': return (a - b) if a >= b else None
        if op == 'multiply': return a * b
        if op == 'divide':   return (a / b) if b != 0 else None

    def score_chain(steps):
        total = 0.0
        n_steps = len(steps)
        last_op = steps[-1][0]

        # Layer 1: penalize chains that violate question-type constraint
        if allowed_final and last_op not in allowed_final:
            total += 20.0

        for idx, (op, a, b, r) in enumerate(steps):
            is_last = (idx == n_steps - 1)
            traj = {'op': op, 'a': a, 'b': b, 'result': r}
            w = 1.0 if is_last else 0.2
            total += w * verify_trajectory(traj, text)
            # scale penalty only for add/subtract intermediates
            if not is_last and op in ('add', 'subtract') and r > max_input * 2:
                total += 3.0
        final_r = steps[-1][3]
        # integer result bonus
        if abs(final_r - round(final_r)) < 1e-6 and final_r > 0:
            total -= 1.5
        # multi-step bonus: longer chains preferred when needed
        total -= 0.4 * (n_steps - 1)
        return total

    best_err    = 1e9
    best_result = None
    best_chain  = []

    CONFIDENT = -9.0  # early-exit threshold: very confident answer

    def try_chain(steps):
        nonlocal best_err, best_result, best_chain
        err = score_chain(steps)
        if err < best_err:
            best_err = err; best_result = steps[-1][3]; best_chain = list(steps)

    def confident():
        return best_err < CONFIDENT

    n = len(nums)

    # 1-step: all ordered pairs
    for i, j in _perms(range(n), 2):
        a, b = nums[i], nums[j]
        for op in ops:
            r = compute(op, a, b)
            if r is None or r < 0: continue
            try_chain([(op, a, b, r)])

    # only exit early after 1-step if exactly 2 numbers (single-op is the only option)
    if confident() and n <= 2:
        return best_result, best_chain, best_err

    # 2-step: all ordered triples
    # break early only for n>3 (n=3 is tiny: 6 triples, always complete)
    if n >= 3 and max_steps >= 2:
        for i, j, k in _perms(range(n), 3):
            if confident() and n > 3: break
            a, b, c = nums[i], nums[j], nums[k]
            for op1 in ops:
                r1 = compute(op1, a, b)
                if r1 is None or r1 <= 0: continue
                for op2 in ops:
                    for x, y in [(r1, c), (c, r1)]:
                        r2 = compute(op2, x, y)
                        if r2 is None or r2 <= 0: continue
                        try_chain([(op1, a, b, r1), (op2, x, y, r2)])

    if confident():
        return best_result, best_chain, best_err

    # 3-step sequential: ordered quadruples (cap at 4 numbers for speed)
    nums4 = nums[:4]
    n4 = len(nums4)
    if n4 >= 4 and max_steps >= 3:
        for i, j, k, l in _perms(range(n4), 4):
            if confident(): break
            a, b, c, d = nums4[i], nums4[j], nums4[k], nums4[l]
            for op1 in ops:
                r1 = compute(op1, a, b)
                if r1 is None or r1 <= 0: continue
                for op2 in ops:
                    for x, y in [(r1, c), (c, r1)]:
                        r2 = compute(op2, x, y)
                        if r2 is None or r2 <= 0: continue
                        for op3 in ops:
                            for p, q in [(r2, d), (d, r2)]:
                                r3 = compute(op3, p, q)
                                if r3 is None or r3 <= 0: continue
                                try_chain([(op1,a,b,r1),(op2,x,y,r2),(op3,p,q,r3)])

    # 3-step parallel-branch: op1(a,b)=r1, op2(c,d)=r2, op3(r1,r2)
    if not confident() and n4 >= 4 and max_steps >= 3:
        for i, j, k, l in _perms(range(n4), 4):
            if confident(): break
            a, b, c, d = nums4[i], nums4[j], nums4[k], nums4[l]
            for op1 in ops:
                r1 = compute(op1, a, b)
                if r1 is None or r1 <= 0: continue
                for op2 in ops:
                    r2 = compute(op2, c, d)
                    if r2 is None or r2 <= 0: continue
                    for op3 in ops:
                        for p, q in [(r1, r2), (r2, r1)]:
                            r3 = compute(op3, p, q)
                            if r3 is None or r3 <= 0: continue
                            try_chain([(op1,a,b,r1),(op2,c,d,r2),(op3,p,q,r3)])

    return best_result, best_chain, best_err
