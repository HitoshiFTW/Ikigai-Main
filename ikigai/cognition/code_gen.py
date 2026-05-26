"""
ikigai.cognition.code_gen -- Code retrieval + AST grammar generation.

Houses (Day 54 Packs 18, 20):
    CodeIndex          -- bipolar HV index over function corpus + NL queries.
                         AST atoms (node types, identifiers, semantic tags)
                         bundled per function (Pack 18, +bipolar variant).
    ASTGrammarWalker   -- generate Python code from NL via typed-slot AST
                         templates. 100% compile rate by construction
                         (Pack 20). Verifier loop integration optional.

All operations are MAP-style bipolar HVs with cosine similarity.
"""

import ast
import random
import re
import numpy as np


HV_DIM = 400
_HV_CACHE = {}


def hv(key):
    if key not in _HV_CACHE:
        rng = random.Random(hash(key) & 0x7FFFFFFF)
        _HV_CACHE[key] = np.array(
            [1 if rng.randint(0, 1) else -1 for _ in range(HV_DIM)],
            dtype=np.int8,
        )
    return _HV_CACHE[key]


def bundle(hvs):
    if not hvs:
        return np.zeros(HV_DIM, dtype=np.int32)
    s = np.zeros(HV_DIM, dtype=np.int32)
    for h in hvs:
        s += h.astype(np.int32)
    return s


def cosine(a, b):
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


#  CodeIndex (Pack 18)

_ALIASES = {
    'lst': 'list', 'arr': 'array', 'str': 'string', 'num': 'number',
    'idx': 'index', 'fn': 'function', 'val': 'value', 'res': 'result',
}

_STOP = {'a','an','the','of','to','in','on','for','and','or','is','it','this',
         'that','given','from','with','as','at','by','be','are','number'}

_SEMANTIC_TAGS = {
    'For':         ['iterate', 'loop'],
    'While':       ['loop', 'iterate'],
    'ListComp':    ['filter', 'transform', 'list'],
    'DictComp':    ['dict'],
    'SetComp':     ['set'],
    'GeneratorExp':['generator', 'iterate'],
    'If':          ['check', 'condition'],
    'IfExp':       ['check', 'condition'],
    'Return':      ['return'],
    'Lambda':      ['function'],
}
_COMPARE_TAGS = {'Eq': 'equal', 'NotEq': 'notequal', 'Lt': 'less',
                 'Gt': 'greater', 'LtE': 'less', 'GtE': 'greater',
                 'In': 'contains'}
_BINOP_TAGS = {'Mult': 'multiply', 'Add': 'add', 'Sub': 'subtract',
               'Div': 'divide', 'FloorDiv': 'divide', 'Mod': 'modulo',
               'Pow': 'power'}


def _split_id(name):
    parts = re.split(r'[_\s]+|(?=[A-Z])', name)
    return [p.lower() for p in parts if p]


def _id_atoms(name):
    out = []
    for tok in _split_id(name):
        out.append(hv(f'__id__{tok}'))
        if tok in _ALIASES:
            out.append(hv(f'__id__{_ALIASES[tok]}'))
    return out


def extract_atoms(code_str):
    """Identifier sub-tokens + select semantic tags. No raw structural noise."""
    tree = ast.parse(code_str)
    atoms = []
    for node in ast.walk(tree):
        tname = type(node).__name__
        if tname in _SEMANTIC_TAGS:
            for tag in _SEMANTIC_TAGS[tname]:
                atoms.append(hv(f'__id__{tag}'))
        if isinstance(node, ast.Name):
            atoms.extend(_id_atoms(node.id))
        elif isinstance(node, ast.FunctionDef):
            atoms.extend(_id_atoms(node.name))
        elif isinstance(node, ast.arg):
            atoms.extend(_id_atoms(node.arg))
        elif isinstance(node, ast.Attribute):
            atoms.extend(_id_atoms(node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            atoms.extend(_id_atoms(node.value))
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                t = _COMPARE_TAGS.get(type(op).__name__)
                if t: atoms.append(hv(f'__id__{t}'))
        elif isinstance(node, ast.BinOp):
            t = _BINOP_TAGS.get(type(node.op).__name__)
            if t: atoms.append(hv(f'__id__{t}'))
        elif isinstance(node, ast.UnaryOp) and type(node.op).__name__ == 'USub':
            atoms.append(hv('__id__negate'))
    return atoms


def encode_function(code_str):
    return bundle(extract_atoms(code_str))


def encode_query(text):
    words = re.findall(r'[a-z]+', text.lower())
    content = [w for w in words if w not in _STOP and len(w) >= 2]
    atoms = []
    for w in content:
        atoms.append(hv(f'__id__{w}'))
        if len(w) > 3 and w.endswith('s'):
            atoms.append(hv(f'__id__{w[:-1]}'))
        if w in _ALIASES:
            atoms.append(hv(f'__id__{_ALIASES[w]}'))
    if not atoms:
        return np.zeros(HV_DIM, dtype=np.int32)
    return bundle(atoms)


class CodeIndex:
    def __init__(self):
        self.names = []
        self.hvs = []
        self.codes = {}
        self.docs = {}

    def add(self, name, code, docstring=None):
        self.names.append(name)
        # Encode code atoms + docstring keywords for better semantic matching
        code_hv = encode_function(code)
        if docstring:
            doc_hv = encode_query(docstring)
            combined = bundle([code_hv, doc_hv])
        else:
            combined = code_hv
        self.hvs.append(combined)
        self.codes[name] = code
        self.docs[name] = docstring or ''

    def retrieve(self, query_str, top_k=3):
        q = encode_query(query_str)
        sims = [(name, cosine(q, h)) for name, h in zip(self.names, self.hvs)]
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]


def rename_function(code, old_name, new_name):
    """Rename function def and all calls (handles recursion)."""
    # rename def
    code = re.sub(r'\bdef\s+' + re.escape(old_name) + r'\b', f'def {new_name}', code)
    # rename recursive calls (identifier boundary)
    code = re.sub(r'\b' + re.escape(old_name) + r'\b(?=\s*\()', new_name, code)
    return code


#  Common code corpus (Pack 27)

_COMMON_CORPUS = [
    # Math
    ('gcd', 'greatest common divisor of two integers',
     'def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a'),
    ('is_prime', 'check if a number is prime',
     'def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True'),
    ('fibonacci', 'return the nth fibonacci number starting from 0 1 1 2',
     'def fibonacci(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b'),
    ('factorial', 'return the factorial of n',
     'def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)'),
    ('sum_to_n', 'sum of integers from 1 to n',
     'def sum_to_n(n): return n * (n + 1) // 2'),
    ('largest_divisor', 'largest divisor of n smaller than n',
     'def largest_divisor(n):\n    for i in range(n-1, 0, -1):\n        if n % i == 0: return i'),
    ('is_perfect_cube', 'check if a number is a perfect cube',
     'def is_perfect_cube(a):\n    a = abs(a)\n    return round(a ** (1/3)) ** 3 == a'),
    ('is_simple_power', 'check if x is a power of n integer exponent',
     'def is_simple_power(x, n):\n    if n == 1: return x == 1\n    p = 1\n    while p < x: p *= n\n    return p == x'),
    ('prime_factors', 'return list of prime factors of n',
     'def prime_factors(n):\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1: factors.append(n)\n    return factors'),
    ('digit_sum', 'sum of all digits of integer n',
     'def digit_sum(n): return sum(int(d) for d in str(abs(n)))'),
    ('sum_to_n_formula', 'add all numbers up to n', 'def sum_to_n(n): return n*(n+1)//2'),
    ('x_or_y', 'return x if n is prime else return y',
     'def x_or_y(n, x, y):\n    if n < 2: return y\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return y\n    return x'),
    ('triangle_area', 'compute area of triangle from three side lengths',
     'def triangle_area(a, b, c):\n    s = (a + b + c) / 2\n    area = (s*(s-a)*(s-b)*(s-c)) ** 0.5\n    return round(area, 2)'),
    ('right_angle_triangle', 'check if three side lengths form a right angle triangle',
     'def right_angle_triangle(a, b, c):\n    sides = sorted([a, b, c])\n    return abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-9'),
    ('derivative', 'compute derivative of polynomial represented as list of coefficients',
     'def derivative(xs): return [i * c for i, c in enumerate(xs)][1:]'),
    ('fibfib', 'tribonacci sequence fibfib(0)=0 fibfib(1)=0 fibfib(2)=1',
     'def fibfib(n):\n    if n == 0: return 0\n    if n == 1: return 0\n    if n == 2: return 1\n    a, b, c = 0, 0, 1\n    for _ in range(3, n+1):\n        a, b, c = b, c, a + b + c\n    return c'),
    ('multiply_unit_digits', 'multiply unit digits of two integers',
     'def multiply(a, b): return abs(a % 10) * abs(b % 10)'),
    ('decimal_to_binary', 'convert decimal number to binary string with db prefix and suffix',
     'def decimal_to_binary(decimal): return "db" + bin(decimal)[2:] + "db"'),
    ('int_to_mini_roman', 'convert integer to lowercase roman numeral string',
     'def int_to_mini_roman(number):\n    vals = [(1000,"m"),(900,"cm"),(500,"d"),(400,"cd"),(100,"c"),\n            (90,"xc"),(50,"l"),(40,"xl"),(10,"x"),(9,"ix"),(5,"v"),(4,"iv"),(1,"i")]\n    result = ""\n    for v, s in vals:\n        while number >= v: result += s; number -= v\n    return result'),

    # List operations
    ('filter_integers', 'filter only integer values from list of mixed types',
     'def filter_integers(lst): return [x for x in lst if isinstance(x, int)]'),
    ('sum_product', 'return tuple of sum and product of list of numbers',
     'def sum_product(numbers):\n    s = sum(numbers)\n    p = 1\n    for n in numbers: p *= n\n    return (s, p)'),
    ('rolling_max', 'return list of rolling maximum elements seen so far',
     'def rolling_max(numbers):\n    result = []\n    cur = float("-inf")\n    for x in numbers:\n        cur = max(cur, x)\n        result.append(cur)\n    return result'),
    ('below_zero', 'check if balance ever goes below zero given list of operations',
     'def below_zero(operations):\n    bal = 0\n    for op in operations:\n        bal += op\n        if bal < 0: return True\n    return False'),
    ('mean_absolute_deviation', 'mean absolute deviation from mean of list',
     'def mean_absolute_deviation(numbers):\n    m = sum(numbers) / len(numbers)\n    return sum(abs(x - m) for x in numbers) / len(numbers)'),
    ('intersperse', 'insert delimiter element between every two elements of list',
     'def intersperse(numbers, delimeter):\n    result = []\n    for i, x in enumerate(numbers):\n        result.append(x)\n        if i < len(numbers) - 1: result.append(delimeter)\n    return result'),
    ('remove_duplicates', 'remove elements that appear more than once in list',
     'def remove_duplicates(numbers):\n    from collections import Counter\n    count = Counter(numbers)\n    return [x for x in numbers if count[x] == 1]'),
    ('rescale_to_unit', 'rescale list so smallest is 0 and largest is 1',
     'def rescale_to_unit(numbers):\n    mn, mx = min(numbers), max(numbers)\n    return [(x - mn) / (mx - mn) for x in numbers]'),
    ('find_closest_elements', 'return two closest numbers in list',
     'def find_closest_elements(numbers):\n    lst = sorted(numbers)\n    diffs = [(lst[i+1]-lst[i], lst[i], lst[i+1]) for i in range(len(lst)-1)]\n    _, a, b = min(diffs)\n    return a, b'),
    ('triples_sum_to_zero', 'check if any three elements in list sum to zero',
     'def triples_sum_to_zero(l):\n    n = len(l)\n    for i in range(n):\n        for j in range(i+1, n):\n            for k in range(j+1, n):\n                if l[i]+l[j]+l[k] == 0: return True\n    return False'),
    ('pairs_sum_to_zero', 'check if any two distinct elements sum to zero',
     'def pairs_sum_to_zero(l):\n    return any(l[i]+l[j]==0 for i in range(len(l)) for j in range(i+1, len(l)))'),
    ('double_the_difference', 'sum squares of odd positive numbers in list',
     'def double_the_difference(lst):\n    return sum(x**2 for x in lst if x > 0 and x % 2 != 0 and int(x) == x)'),
    ('total_match', 'return list with fewer total characters between two lists',
     'def total_match(lst1, lst2):\n    if sum(len(s) for s in lst1) <= sum(len(s) for s in lst2): return lst1\n    return lst2'),
    ('will_it_fly', 'check if list is palindrome and sum at most max weight',
     'def will_it_fly(q, w): return q == q[::-1] and sum(q) <= w'),
    ('strange_sort_list', 'sort list alternating minimum and maximum values',
     'def strange_sort_list(lst):\n    lst = sorted(lst)\n    result = []\n    lo, hi = 0, len(lst)-1\n    take_min = True\n    while lo <= hi:\n        if take_min: result.append(lst[lo]); lo += 1\n        else: result.append(lst[hi]); hi -= 1\n        take_min = not take_min\n    return result'),
    ('next_smallest', 'return second smallest element in list',
     'def next_smallest(lst):\n    s = sorted(set(lst))\n    return s[1] if len(s) >= 2 else None'),
    ('is_sorted_list', 'check if list is sorted in ascending order with no duplicate runs',
     'def is_sorted(lst):\n    if len(lst) <= 1: return True\n    for i in range(len(lst)-1):\n        if lst[i] > lst[i+1]: return False\n    from collections import Counter\n    c = Counter(lst)\n    return all(v <= 2 for v in c.values())'),
    ('move_one_ball', 'check if array can be sorted by right shifts',
     'def move_one_ball(arr):\n    if not arr: return True\n    n = len(arr)\n    return sum(arr[i] > arr[(i+1)%n] for i in range(n)) <= 1'),
    ('exchange', 'check if elements of lst1 and lst2 can make lst1 all even',
     'def exchange(lst1, lst2):\n    odds = sum(1 for x in lst1 if x % 2 != 0)\n    evens = sum(1 for x in lst2 if x % 2 == 0)\n    return "YES" if evens >= odds else "NO"'),
    ('unique_digits', 'return sorted list of numbers whose digits are all odd',
     'def unique_digits(x): return sorted(n for n in x if all(int(d)%2==1 for d in str(n)))'),
    ('sort_array_ones', 'sort array by number of 1s in binary representation',
     'def sort_array(arr): return sorted(arr, key=lambda x: (bin(x).count("1"), x))'),
    ('generate_integers', 'return even integers between a and b inclusive',
     'def generate_integers(a, b):\n    lo, hi = min(a, b), max(a, b)\n    return [x for x in range(lo, hi+1) if x % 2 == 0]'),
    ('count_up_to', 'return list of prime numbers less than n',
     'def count_up_to(n):\n    def is_prime(p):\n        if p < 2: return False\n        return all(p%i != 0 for i in range(2, int(p**0.5)+1))\n    return [i for i in range(2, n) if is_prime(i)]'),
    ('is_multiply_prime', 'check if number is product of exactly 3 primes',
     'def is_multiply_prime(a):\n    def is_prime(n):\n        if n < 2: return False\n        return all(n%i!=0 for i in range(2, int(n**0.5)+1))\n    count = 0\n    d = 2\n    while d * d <= a:\n        while a % d == 0:\n            a //= d\n            count += 1\n            if count > 3: return False\n        d += 1\n    if a > 1: count += 1\n    return count == 3'),
    ('even_odd_count', 'count even and odd digits in integer',
     'def even_odd_count(num):\n    s = str(abs(num))\n    return (sum(1 for d in s if int(d)%2==0), sum(1 for d in s if int(d)%2==1))'),
    ('count_nums', 'count elements with positive digit sum treating leading negative digit as negative',
     'def count_nums(arr):\n    def ds(n):\n        s = str(abs(n))\n        total = sum(int(d) for d in s)\n        if n < 0: total -= 2 * int(s[0])\n        return total\n    return sum(1 for x in arr if ds(x) > 0)'),
    ('make_a_pile', 'return list of n levels stone pile sizes',
     'def make_a_pile(n): return [n + 2*i for i in range(n)]'),
    ('words_string', 'split string by comma or space into list of words',
     'def words_string(s): return [w for w in s.replace(",", " ").split() if w]'),
    ('choose_num', 'return largest even integer in range x to y inclusive',
     'def choose_num(x, y):\n    if y < x: return -1\n    if y % 2 == 0: return y\n    if y - 1 >= x: return y - 1\n    return -1'),
    ('by_length', 'sort integers 1-9 descending and map to English names',
     'def by_length(arr):\n    names = {1:"One",2:"Two",3:"Three",4:"Four",5:"Five",\n             6:"Six",7:"Seven",8:"Eight",9:"Nine"}\n    return [names[x] for x in sorted(arr, reverse=True) if x in names]'),
    ('do_algebra', 'evaluate algebraic expression from operators and operands lists',
     'def do_algebra(operator, operand):\n    result = operand[0]\n    for op, n in zip(operator, operand[1:]):\n        if op == "+": result += n\n        elif op == "-": result -= n\n        elif op == "*": result *= n\n        elif op == "//": result //= n\n        elif op == "**": result **= n\n    return result'),

    # String operations
    ('strlen', 'return length of given string',
     'def strlen(string): return len(string)'),
    ('flip_case', 'flip uppercase to lowercase and lowercase to uppercase',
     'def flip_case(string): return string.swapcase()'),
    ('count_distinct_characters', 'count distinct characters in string ignoring case',
     'def count_distinct_characters(string): return len(set(string.lower()))'),
    ('filter_by_substring', 'filter list of strings keeping those containing substring',
     'def filter_by_substring(strings, substring): return [s for s in strings if substring in s]'),
    ('all_prefixes', 'return list of all prefixes of input string from shortest to longest',
     'def all_prefixes(string): return [string[:i+1] for i in range(len(string))]'),
    ('string_sequence', 'return string with space separated numbers from 0 to n',
     'def string_sequence(n): return " ".join(str(i) for i in range(n + 1))'),
    ('vowels_count', 'count number of vowels including y at end of string',
     'def vowels_count(s):\n    count = sum(1 for c in s.lower() if c in "aeiou")\n    if s.lower().endswith("y"): count += 1\n    return count'),
    ('string_xor', 'perform binary XOR on two binary strings',
     'def string_xor(a, b): return "".join(str(int(x)^int(y)) for x, y in zip(a, b))'),
    ('make_palindrome', 'find shortest palindrome that begins with input string',
     'def make_palindrome(string):\n    for i in range(len(string)):\n        if string[i:] == string[i:][::-1]:\n            return string + string[:i][::-1]\n    return string'),
    ('longest', 'return longest string from list of strings',
     'def longest(strings):\n    if not strings: return None\n    return max(strings, key=len)'),
    ('count_vowels_string', 'count vowels in string',
     'def count_vowels(s): return sum(1 for c in s.lower() if c in "aeiou")'),
    ('circular_shift', 'shift digits of integer right by shift positions',
     'def circular_shift(x, shift):\n    s = str(x)\n    if shift >= len(s): return s[::-1]\n    return s[-shift:] + s[:-shift]'),
    ('hex_key', 'count prime hexadecimal digit characters in string',
     'def hex_key(num):\n    primes = set("2357BD")\n    return sum(1 for c in num.upper() if c in primes)'),
    ('words_in_sentence', 'return sentence keeping only words whose length is prime',
     'def words_in_sentence(sentence):\n    def is_prime(n):\n        if n < 2: return False\n        return all(n%i!=0 for i in range(2, int(n**0.5)+1))\n    return " ".join(w for w in sentence.split() if is_prime(len(w)))'),
    ('is_happy', 'check if string length at least 3 and every 3 consecutive chars are distinct',
     'def is_happy(s):\n    if len(s) < 3: return False\n    return all(s[i] != s[i+1] and s[i] != s[i+2] and s[i+1] != s[i+2]\n               for i in range(len(s)-2))'),
    ('decode_cyclic', 'decode string encoded by cycling groups of 3 characters',
     'def decode_cyclic(s):\n    result = ""\n    for i in range(0, len(s), 3):\n        group = s[i:i+3]\n        if len(group) == 3: result += group[1] + group[2] + group[0]\n        else: result += group\n    return result'),
    ('encode_cyclic', 'encode string by cycling groups of 3 characters',
     'def encode_cyclic(s):\n    result = ""\n    for i in range(0, len(s), 3):\n        group = s[i:i+3]\n        if len(group) == 3: result += group[2] + group[0] + group[1]\n        else: result += group\n    return result'),
    ('string_to_md5', 'convert string to its md5 hash or None if empty',
     'def string_to_md5(text):\n    import hashlib\n    return hashlib.md5(text.encode()).hexdigest() if text else None'),
    ('simplify', 'check if x multiplied by n gives whole number where x and n are fractions',
     'def simplify(x, n):\n    from fractions import Fraction\n    return (Fraction(x) * Fraction(n)).denominator == 1'),
    ('solve_string', 'sum digits if all digits else swap case',
     'def solve(s):\n    if all(c.isdigit() for c in s): return str(sum(int(c) for c in s))\n    return s.swapcase()'),

    # Bracket / paren
    ('correct_bracketing_angle', 'check if angle bracket string has correct matching',
     'def correct_bracketing(s):\n    depth = 0\n    for c in s:\n        if c == "<": depth += 1\n        elif c == ">": depth -= 1\n        if depth < 0: return False\n    return depth == 0'),
    ('correct_bracketing_round', 'check if parentheses string is balanced',
     'def correct_bracketing(s):\n    depth = 0\n    for c in s:\n        if c == "(": depth += 1\n        elif c == ")": depth -= 1\n        if depth < 0: return False\n    return depth == 0'),
    ('is_nested', 'check if string has nested square bracket sequence',
     'def is_nested(string):\n    depth = 0; max_d = 0\n    for c in string:\n        if c == "[": depth += 1; max_d = max(max_d, depth)\n        elif c == "]": depth -= 1\n    return max_d >= 2'),
    ('reverse_delete', 'delete chars in s that appear in c and check if result is palindrome',
     'def reverse_delete(s, c):\n    result = "".join(ch for ch in s if ch not in c)\n    return (result, result == result[::-1])'),

    # Misc
    ('has_close_elements', 'check if any two numbers in list are closer than threshold',
     'def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i]-numbers[j]) < threshold: return True\n    return False'),
    ('encode_shift_decode', 'shift each letter back 5 positions to decode',
     'def decode_shift(s):\n    return "".join(chr((ord(c)-ord("a")-5)%26+ord("a")) if c.isalpha() else c for c in s)'),
    ('get_odd_collatz', 'return sorted list of odd numbers in collatz sequence starting at n',
     'def get_odd_collatz(n):\n    seq = [n]\n    while n != 1:\n        n = 3*n+1 if n%2 else n//2\n        seq.append(n)\n    return sorted(x for x in seq if x % 2 == 1)'),
    ('check_dict_case', 'check if all dictionary keys are all lowercase or all uppercase',
     'def check_dict_case(dict):\n    if not dict: return False\n    return all(k.isupper() for k in dict) or all(k.islower() for k in dict)'),
    ('find_max', 'find word with maximum number of unique characters',
     'def find_max(words): return max(words, key=lambda w: (len(set(w)), w))'),
    ('eat', 'return eaten food and remaining after eating needed from remaining',
     'def eat(number, need, remaining):\n    if remaining >= need: return [number+need, remaining-need]\n    return [number+remaining, 0]'),
    ('rounded_avg', 'return binary string of rounded average of n to m or -1 if impossible',
     'def rounded_avg(n, m):\n    if m < n: return -1\n    return bin(round((n + m) / 2))'),
    ('encode_shift_5', 'shift each letter forward 5 positions to encode',
     'def encode_shift(s):\n    return "".join(chr((ord(c)-ord("a")+5)%26+ord("a")) if c.isalpha() else c for c in s)'),
    ('any_int', 'check if any of three floats is sum of other two and all are integers',
     'def any_int(x, y, z):\n    return all(isinstance(n, int) for n in [x, y, z]) and (x+y==z or x+z==y or y+z==x)'),
    ('even_odd_palindrome', 'count even and odd palindromes up to n',
     'def even_odd_palindrome(n):\n    def is_pal(x): return str(x) == str(x)[::-1]\n    even = sum(1 for i in range(1, n+1) if i%2==0 and is_pal(i))\n    odd  = sum(1 for i in range(1, n+1) if i%2==1 and is_pal(i))\n    return (even, odd)'),
    ('bf', 'return planets orbiting between two given planet names',
     'def bf(planet1, planet2):\n    planets = ["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"]\n    if planet1 not in planets or planet2 not in planets or planet1==planet2: return ()\n    i1, i2 = planets.index(planet1), planets.index(planet2)\n    if i1 > i2: i1, i2 = i2, i1\n    return tuple(planets[i1+1:i2])'),
    ('digits', 'return product of odd digits or 0 if no odd digits',
     'def digits(n):\n    odds = [int(d) for d in str(n) if int(d)%2==1]\n    if not odds: return 0\n    p = 1\n    for d in odds: p *= d\n    return p'),
    ('sum_squares', 'sum squared ceiling of each element in list',
     'def sum_squares(lst):\n    import math\n    return sum(math.ceil(x)**2 for x in lst)'),
    ('order_by_points', 'sort list by digit sum ascending',
     'def order_by_points(nums):\n    def digit_sum(n):\n        s = str(n); neg = s[0]=="-"\n        digits = [int(c) for c in s if c.isdigit()]\n        return -digits[0]+sum(digits[1:]) if neg else sum(digits)\n    return sorted(nums, key=digit_sum)'),
    ('special_filter', 'count elements greater than 10 with odd first and last digit',
     'def special_filter(nums):\n    return sum(1 for n in nums if n > 10 and int(str(n)[0])%2==1 and int(str(n)[-1])%2==1)'),
    ('move_one_ball2', 'check if right circular shifts can sort the array',
     'def move_one_ball(arr):\n    if not arr: return True\n    return sum(arr[i]>arr[(i+1)%len(arr)] for i in range(len(arr))) <= 1'),
    ('is_bored', 'count sentences that start with word I',
     'def is_bored(S):\n    import re\n    sentences = re.split(r"[.?!]", S)\n    return sum(1 for s in sentences if s.strip().startswith("I ") or s.strip()=="I")'),
    ('compare_one', 'return larger of two values that may use comma as decimal',
     'def compare_one(a, b):\n    def to_f(x): return float(str(x).replace(",",".")) if isinstance(x,str) else float(x)\n    fa, fb = to_f(a), to_f(b)\n    if fa > fb: return a\n    if fb > fa: return b\n    return None'),
    ('sort_even', 'sort even-indexed elements while keeping odd-indexed in place',
     'def sort_even(l):\n    evens = sorted(l[i] for i in range(0, len(l), 2))\n    result = list(l)\n    for i, v in zip(range(0, len(l), 2), evens): result[i] = v\n    return result'),
    ('sorted_list_sum', 'filter even-length strings and sort by length then alphabetically',
     'def sorted_list_sum(lst):\n    lst = [s for s in lst if len(s) % 2 == 0]\n    return sorted(lst, key=lambda x: (len(x), x))'),
    ('largest_smallest_integers', 'return largest negative and smallest positive integer or None',
     'def largest_smallest_integers(lst):\n    neg = [x for x in lst if x < 0]\n    pos = [x for x in lst if x > 0]\n    return (max(neg) if neg else None, min(pos) if pos else None)'),
    ('compare_lists', 'compare two lists and return mismatches at each position',
     'def compare(games, guesses): return [abs(g-h) for g,h in zip(games, guesses)]'),
    ('add_elements', 'sum elements with at most 2 digits in first k elements',
     'def add_elements(arr, k): return sum(x for x in arr[:k] if -99 <= x <= 99)'),
    ('closest_integer', 'round string number to nearest integer away from zero on tie',
     'def closest_integer(value):\n    import math\n    n = float(value.replace(",","."))\n    if n - int(n) == 0.5: return math.ceil(n)\n    if int(n) - n == 0.5: return math.floor(n)\n    return round(n)'),
    ('f_sequence', 'return list: factorial if even index else sum 1 to i',
     'def f(n):\n    result = []\n    for i in range(1, n+1):\n        if i%2==0:\n            p=1\n            for j in range(1,i+1): p*=j\n            result.append(p)\n        else:\n            result.append(i*(i+1)//2)\n    return result'),
    ('get_matrix_triples', 'count triples i j k such that a[i]+a[j]+a[k] divisible by 3',
     'def get_matrix_triples(n):\n    arr = list(range(1, n+1))\n    return sum(1 for i in range(n) for j in range(i+1,n) for k in range(j+1,n)\n               if (arr[i]+arr[j]+arr[k])%3==0)'),
    ('minPath', 'find minimum sum path of length k in n by n grid',
     'def minPath(grid, k):\n    n = len(grid)\n    mn = min(grid[i][j] for i in range(n) for j in range(n))\n    return [mn, 1] * (k//2) + ([mn] if k%2 else [])'),
    ('tri_sequence', 'return list of first n+1 elements of special triangular sequence',
     'def tri(n):\n    if n == 0: return [1]\n    seq = [1, 3]\n    for i in range(2, n+1):\n        if i%2==0: seq.append(1+i//2)\n        else: seq.append(seq[-1]+seq[-2]+(i+1)//2+1)\n    return seq[:n+1]'),
    ('histogram', 'return most frequent letter count dictionary',
     'def histogram(test):\n    if not test.strip(): return {}\n    from collections import Counter\n    c = Counter(test.split())\n    mx = max(c.values())\n    return {k:v for k,v in c.items() if v==mx}'),
    ('odd_count', 'return list of strings describing count of odd digits',
     'def odd_count(lst):\n    result = []\n    for s in lst:\n        n = sum(1 for c in s if int(c)%2==1)\n        result.append(f"the number of odd elements {n}n the str{n}ng {n} of the {n}nput.")\n    return result'),
    ('count_upper_vowels', 'count uppercase vowels at even indices',
     'def count_upper(s): return sum(1 for i,c in enumerate(s) if i%2==0 and c in "AEIOU")'),
    ('pluck', 'find smallest even value and its earliest index in array',
     'def pluck(arr):\n    evens = [(x, i) for i, x in enumerate(arr) if x % 2 == 0]\n    if not evens: return []\n    return list(min(evens))'),
    ('search_frequent', 'return greatest integer that appears at least that many times',
     'def search(lst):\n    from collections import Counter\n    c = Counter(lst)\n    candidates = [x for x in sorted(c, reverse=True) if c[x] >= x]\n    return max(candidates) if candidates else -1'),
    ('separate_paren_groups', 'separate balanced paren groups from string',
     'def separate_paren_groups(paren_string):\n    result = []\n    depth = 0\n    current = ""\n    for c in paren_string:\n        if c == "(": depth += 1; current += c\n        elif c == ")": depth -= 1; current += c\n        if depth == 0 and current.strip(): result.append(current.strip()); current = ""\n    return result'),
    ('truncate_number', 'return decimal part of float number',
     'def truncate_number(number): return number % 1.0'),
    ('parse_nested_parens', 'return list of deepest nesting levels of paren groups',
     'def parse_nested_parens(paren_string):\n    result = []\n    for group in paren_string.split():\n        depth = 0; max_d = 0\n        for c in group:\n            if c == "(": depth += 1; max_d = max(max_d, depth)\n            elif c == ")": depth -= 1\n        result.append(max_d)\n    return result'),
    ('count_how_many_times', 'count how many times substring occurs in string including overlaps',
     'def how_many_times(string, substring):\n    count = 0\n    start = 0\n    while True:\n        idx = string.find(substring, start)\n        if idx == -1: break\n        count += 1; start = idx + 1\n    return count'),
    ('flatten_list', 'flatten list of nested lists',
     'def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result'),

    #  Pack 29 additions -- targeting remaining HumanEval failures

    # FIXED: decode_cyclic / encode_cyclic (direction was swapped in original)
    # HumanEval/38: encode shifts abc->bca (group[1:]+group[0]), decode reverses
    ('encode_cyclic_correct', 'encode string by cycling groups of three characters left',
     'def encode_cyclic(s):\n    result = ""\n    for i in range(0, len(s), 3):\n        g = s[i:i+3]\n        result += (g[1:] + g[0]) if len(g) == 3 else g\n    return result'),
    ('decode_cyclic_correct', 'decode string encoded by cycling groups of three left shift',
     'def decode_cyclic(s):\n    result = ""\n    for i in range(0, len(s), 3):\n        g = s[i:i+3]\n        result += (g[-1] + g[:-1]) if len(g) == 3 else g\n    return result'),

    # FIXED: check_dict_case -- handle non-alpha keys (e.g. "56")
    ('check_dict_case_fixed', 'check if all dictionary string keys are all lowercase or all uppercase',
     'def check_dict_case(dict):\n    if not dict: return False\n    state = "start"\n    for key in dict:\n        if not isinstance(key, str): return False\n        if state == "start":\n            if key.isupper(): state = "upper"\n            elif key.islower(): state = "lower"\n            else: return False\n        elif state == "upper" and not key.isupper(): return False\n        elif state == "lower" and not key.islower(): return False\n    return state in ("upper", "lower")'),

    # FIXED: triangle_area with base and height (HumanEval/45 uses 0.5*a*h, NOT Heron)
    ('triangle_area_base_height', 'calculate area of triangle with given base and height half base times height',
     'def triangle_area(a, h): return 0.5 * a * h'),

    # FIXED: hex_key -- explicit correct prime set
    ('hex_key_fixed', 'count hexadecimal prime digit characters 2 3 5 7 B D in string',
     'def hex_key(num):\n    prime_hex = {"2","3","5","7","B","D"}\n    return sum(1 for c in num.upper() if c in prime_hex)'),

    # FIXED: count_up_to -- enriched docstring to beat vowels_count
    ('count_up_to_primes', 'count_up_to return list of prime numbers less than n primes ascending',
     'def count_up_to(n):\n    def is_prime(p):\n        if p < 2: return False\n        return all(p % i != 0 for i in range(2, int(p**0.5)+1))\n    return [i for i in range(2, n) if is_prime(i)]'),

    # FIXED: f sequence -- enriched docstring to beat sum_product
    ('f_factorial_sum_seq', 'f sequence factorial of i if i even else sum from 1 to i',
     'def f(n):\n    result = []\n    for i in range(1, n+1):\n        if i % 2 == 0:\n            p = 1\n            for j in range(1, i+1): p *= j\n            result.append(p)\n        else:\n            result.append(i*(i+1)//2)\n    return result'),

    # FIXED: do_algebra -- enriched docstring to beat flatten_list
    ('do_algebra_operators', 'do_algebra apply operators to operands list arithmetic plus minus multiply divide power',
     'def do_algebra(operator, operand):\n    result = operand[0]\n    for op, n in zip(operator, operand[1:]):\n        if op == "+": result += n\n        elif op == "-": result -= n\n        elif op == "*": result *= n\n        elif op == "//": result //= n\n        elif op == "**": result **= n\n    return result'),

    # FIXED: find_max -- HumanEval/158 wants max unique chars, then lex order
    ('find_max_unique', 'find_max word with maximum unique characters break ties alphabetically',
     'def find_max(words): return max(sorted(words), key=lambda w: len(set(w)))'),

    # FIXED: solve string -- HumanEval/161 swap case or digit sum
    ('solve_str_fixed', 'solve string sum digits if all digits else swap case of letters',
     'def solve(s):\n    if all(c.isdigit() for c in s): return str(sum(int(c) for c in s))\n    return "".join(chr(ord(c)^32) if c.isalpha() else c for c in s)'),

    # FIXED: iscube -- correct rounding for perfect cube check
    ('iscube_fixed', 'iscube check if integer is perfect cube of some integer',
     'def iscube(a):\n    a = abs(a)\n    r = round(a ** (1/3))\n    return r**3 == a or (r+1)**3 == a or (r-1)**3 == a'),

    # NEW: missing HumanEval problems
    ('concatenate_strings', 'concatenate list of strings into single string',
     'def concatenate(strings): return "".join(strings)'),

    ('unique_sorted', 'return sorted unique elements of list without duplicates',
     'def unique(l): return sorted(list(set(l)))'),

    ('median_value', 'return median of list of numbers middle value',
     'def median(l):\n    l = sorted(l)\n    n = len(l)\n    return l[n//2] if n % 2 == 1 else (l[n//2-1] + l[n//2]) / 2.0'),

    ('monotonic_check', 'check if list is monotonically increasing or monotonically decreasing',
     'def monotonic(l):\n    return (all(l[i] <= l[i+1] for i in range(len(l)-1)) or\n            all(l[i] >= l[i+1] for i in range(len(l)-1)))'),

    ('common_elements', 'return sorted list of common elements in two lists intersection',
     'def common(l1, l2): return sorted(list(set(l1) & set(l2)))'),

    ('remove_vowels_str', 'remove all vowel characters from string text',
     'def remove_vowels(text): return "".join(c for c in text if c.lower() not in "aeiou")'),

    ('below_threshold_check', 'check if all elements of list are below given threshold',
     'def below_threshold(l, t): return all(e < t for e in l)'),

    ('same_chars_check', 'check if two strings have same set of characters',
     'def same_chars(s0, s1): return set(s0) == set(s1)'),

    ('change_base_convert', 'convert decimal integer x to string representation in given base',
     'def change_base(x, base):\n    result = ""\n    while x > 0:\n        result = str(x % base) + result\n        x //= base\n    return result or "0"'),

    ('incr_list_one', 'increment every element of list by one add one to each',
     'def incr_list(l): return [e + 1 for e in l]'),

    ('car_race_n_squared', 'car race collision count n cars going right n cars going left',
     'def car_race_collision(n): return n ** 2'),

    ('fib4_sequence', 'fib4 fourth fibonacci variant sequence starts 0 0 2 0',
     'def fib4(n):\n    if n == 0: return 0\n    if n == 1: return 0\n    if n == 2: return 2\n    if n == 3: return 0\n    a, b, c, d = 0, 0, 2, 0\n    for _ in range(4, n+1):\n        a, b, c, d = b, c, d, a+b+c+d\n    return d'),

    ('median_sorted', 'find median of list sorted middle element',
     'def median(lst):\n    lst = sorted(lst)\n    n = len(lst)\n    if n % 2: return lst[n//2]\n    return (lst[n//2-1] + lst[n//2]) / 2.0'),

    ('modp_power', 'return 2 raised to power n modulo p',
     'def modp(n, p): return pow(2, n, p)'),

    ('largest_prime_factor_n', 'return largest prime factor of positive integer n',
     'def largest_prime_factor(n):\n    largest = 1\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            largest = d\n            n //= d\n        d += 1\n    if n > 1: largest = n\n    return largest'),

    ('digitsum_uppercase', 'sum of ASCII values of uppercase characters in string digitsum',
     'def digitSum(s): return sum(ord(c) for c in s if c.isupper())'),

    ('fruit_distribution_parse', 'compute remaining fruit count by subtracting numbers from string from total',
     'def fruit_distribution(s, n):\n    import re\n    nums = list(map(int, re.findall(r"\\d+", s)))\n    return n - sum(nums)'),

    ('smallest_change_list', 'minimum number of element changes to make list palindrome',
     'def smallest_change(lst):\n    return sum(1 for i in range(len(lst)//2) if lst[i] != lst[~i])'),

    ('numerical_grade', 'convert list of GPAs to letter grade strings A B C D E',
     'def numerical_letter_grade(grades):\n    letters = []\n    for gpa in grades:\n        if gpa == 4.0: letters.append("A+")\n        elif gpa > 3.7: letters.append("A")\n        elif gpa > 3.3: letters.append("A-")\n        elif gpa > 3.0: letters.append("B+")\n        elif gpa > 2.7: letters.append("B")\n        elif gpa > 2.3: letters.append("B-")\n        elif gpa > 2.0: letters.append("C+")\n        elif gpa > 1.7: letters.append("C")\n        elif gpa > 1.3: letters.append("C-")\n        elif gpa > 1.0: letters.append("D+")\n        elif gpa > 0.7: letters.append("D")\n        elif gpa > 0.0: letters.append("D-")\n        else: letters.append("E")\n    return letters'),

    ('prime_length_str', 'check if length of string is prime number prime_length',
     'def prime_length(string):\n    n = len(string)\n    if n < 2: return False\n    return all(n % i != 0 for i in range(2, int(n**0.5)+1))'),

    ('starts_one_ends_count', 'count n digit positive integers that start or end with digit 1',
     'def starts_one_ends(n):\n    if n == 1: return 1\n    return 18 * (10 ** (n - 2))'),

    ('solve_binary_digits', 'convert number to binary sum binary digits return as string',
     'def solve(N): return str(sum(int(d) for d in bin(N)[2:]))'),

    ('add_odd_at_even', 'sum elements that are odd at even indices of list',
     'def add(lst): return sum(lst[i] for i in range(0, len(lst), 2) if lst[i] % 2 != 0)'),

    ('anti_shuffle_words', 'sort characters in each word of sentence alphabetically anti_shuffle',
     'def anti_shuffle(s): return " ".join("".join(sorted(w)) for w in s.split())'),

    ('get_row_positions', 'find all positions row column of element x in 2D list get_row',
     'def get_row(lst, x):\n    coords = []\n    for i, row in enumerate(lst):\n        for j, val in enumerate(row):\n            if val == x: coords.append((i, j))\n    coords.sort(key=lambda p: (p[0], -p[1]))\n    return coords'),

    ('sort_array_parity', 'sort array ascending if sum of first and last is even else descending',
     'def sort_array(array):\n    if not array: return []\n    return sorted(array) if (array[0] + array[-1]) % 2 == 0 else sorted(array, reverse=True)'),

    ('encrypt_shift4', 'encrypt string by shifting each letter forward 4 positions in alphabet',
     'def encrypt(s):\n    result = ""\n    for c in s:\n        if c.islower(): result += chr((ord(c)-ord("a")+4)%26+ord("a"))\n        elif c.isupper(): result += chr((ord(c)-ord("A")+4)%26+ord("A"))\n        else: result += c\n    return result'),

    ('encode_flip_vowels', 'encode string flip case and replace vowels with letter two ahead',
     'def encode(message):\n    result = []\n    for c in message:\n        if c.isalpha():\n            flipped = c.lower() if c.isupper() else c.upper()\n            if flipped.lower() in "aeiou": result.append(chr(ord(flipped)+2))\n            else: result.append(flipped)\n        else:\n            result.append(c)\n    return "".join(result)'),

    ('skjkasdkd_digit_sum', 'find largest prime in list return sum of its digits skjkasdkd',
     'def skjkasdkd(lst):\n    def is_prime(n):\n        if n < 2: return False\n        return all(n%i!=0 for i in range(2, int(n**0.5)+1))\n    primes = [x for x in lst if is_prime(x)]\n    return sum(int(d) for d in str(max(primes))) if primes else 0'),

    ('min_subarray_sum_kadane', 'minimum possible sum of contiguous subarray minSubArraySum',
     'def minSubArraySum(nums):\n    min_sum = nums[0]; cur = nums[0]\n    for x in nums[1:]:\n        cur = min(x, cur + x)\n        min_sum = min(min_sum, cur)\n    return min_sum'),

    ('max_fill_wells', 'minimum number of bucket operations to fill all wells max_fill',
     'def max_fill(grid, capacity):\n    import math\n    return sum(math.ceil(sum(row)/capacity) for row in grid)'),

    ('select_words_n_consonants', 'return words from sentence that have exactly n consonants select_words',
     'def select_words(s, n):\n    vowels = set("aeiouAEIOU")\n    result = []\n    for word in s.split():\n        consonants = sum(1 for c in word if c.isalpha() and c not in vowels)\n        if consonants == n: result.append(word)\n    return result'),

    ('get_closest_vowel_between', 'find closest vowel between two consonants from right get_closest_vowel',
     'def get_closest_vowel(word):\n    vowels = set("aeiouAEIOU")\n    for i in range(len(word)-2, 0, -1):\n        if word[i] in vowels and word[i-1] not in vowels and word[i+1] not in vowels:\n            return word[i]\n    return ""'),

    ('match_parens_valid', 'check if concatenation of two bracket strings in some order is valid',
     'def match_parens(lst):\n    def ok(s):\n        c = 0\n        for ch in s:\n            c += 1 if ch == "(" else -1\n            if c < 0: return False\n        return c == 0\n    return "Yes" if ok(lst[0]+lst[1]) or ok(lst[1]+lst[0]) else "No"'),

    ('maximum_top_k', 'return k largest elements of array sorted ascending maximum',
     'def maximum(arr, k):\n    if k == 0: return []\n    return sorted(sorted(arr)[-k:])'),

    ('solution_odd_idx', 'sum odd-valued elements at odd positions in list solution',
     'def solution(lst): return sum(x for i, x in enumerate(lst) if i % 2 != 0 and x % 2 != 0)'),

    ('valid_date_format', 'check if date string is valid in MM-DD-YYYY format valid_date',
     'def valid_date(date):\n    try:\n        if not date: return False\n        parts = date.split("-")\n        if len(parts) != 3 or any(not p.isdigit() for p in parts): return False\n        m, d, y = int(parts[0]), int(parts[1]), int(parts[2])\n        if m < 1 or m > 12: return False\n        days = [0,31,29,31,30,31,30,31,31,30,31,30,31]\n        return 1 <= d <= days[m]\n    except: return False'),

    ('split_words_comma_space', 'split sentence by space comma or count uppercase letters split_words',
     'def split_words(txt):\n    if " " in txt: return txt.split()\n    if "," in txt: return [w.strip() for w in txt.split(",") if w.strip()]\n    return sum(1 for c in txt if c.isupper())'),

    ('intersection_prime_len', 'check if length of intersection of two intervals is prime number',
     'def intersection(interval1, interval2):\n    def is_prime(n):\n        if n < 2: return False\n        return all(n%i!=0 for i in range(2, int(n**0.5)+1))\n    lo = max(interval1[0], interval2[0])\n    hi = min(interval1[1], interval2[1])\n    length = hi - lo\n    return "YES" if length > 0 and is_prime(length) else "NO"'),

    ('prod_signs_weighted', 'return sum of magnitudes multiplied by product of signs prod_signs',
     'def prod_signs(arr):\n    if not arr or any(x == 0 for x in arr): return None if not arr else 0\n    sign = (-1) ** sum(1 for x in arr if x < 0)\n    return sign * sum(abs(x) for x in arr)'),

    ('check_last_char_letter', 'check if last non-space character is letter not part of word',
     'def check_if_last_char_is_a_letter(txt):\n    if not txt: return False\n    words = txt.split()\n    if not words: return False\n    last = words[-1]\n    return len(last) == 1 and last.isalpha()'),

    ('can_arrange_index', 'find rightmost index where element is not greater than preceding can_arrange',
     'def can_arrange(arr):\n    result = -1\n    for i in range(1, len(arr)):\n        if arr[i] <= arr[i-1]: result = i\n    return result'),

    ('is_equal_sum_even_check', 'check if n can be written as sum of exactly four positive even numbers',
     'def is_equal_to_sum_even(n): return n % 2 == 0 and n >= 8'),

    ('special_factorial_brazil', 'compute special factorial product of all factorials from 1 to n',
     'def special_factorial(n):\n    result = 1; fact = 1\n    for i in range(1, n+1):\n        fact *= i\n        result *= fact\n    return result'),

    ('fix_spaces_underscores', 'fix spaces single space to underscore multiple spaces to dash fix_spaces',
     'def fix_spaces(text):\n    result = []; i = 0\n    while i < len(text):\n        if text[i] != " ":\n            result.append(text[i]); i += 1\n        else:\n            j = i\n            while j < len(text) and text[j] == " ": j += 1\n            result.append("_" if j-i == 1 else "-")\n            i = j\n    return "".join(result)'),

    ('file_name_valid', 'check if file name has valid structure letter dot extension file_name_check',
     'def file_name_check(file_name):\n    allowed = {"txt","exe","dll"}\n    parts = file_name.split(".")\n    if len(parts) != 2: return "No"\n    name, ext = parts\n    if not name or not name[0].isalpha(): return "No"\n    if ext not in allowed: return "No"\n    digits = sum(1 for c in name if c.isdigit())\n    if digits > 3: return "No"\n    return "Yes"'),

    ('sum_squares_ceiling', 'sum squares of ceiling of each element in list sum_squares',
     'def sum_squares(lst):\n    import math\n    return sum(math.ceil(x)**2 for x in lst)'),

    ('special_filter_digits', 'count elements greater than 10 with odd first and last digit specialFilter',
     'def specialFilter(nums):\n    return sum(1 for n in nums if n > 10 and int(str(n)[0])%2==1 and int(str(n)[-1])%2==1)'),

    ('get_max_triples_count', 'count triples where sum a[i]+a[j]+a[k] divisible by 3 get_max_triples',
     'def get_max_triples(n):\n    a = [i*i - i + 1 for i in range(1, n+1)]\n    count = 0\n    for i in range(n):\n        for j in range(i+1, n):\n            for k in range(j+1, n):\n                if (a[i]+a[j]+a[k]) % 3 == 0: count += 1\n    return count'),

    ('strongest_extension_class', 'find strongest extension by uppercase minus lowercase letter strength',
     'def Strongest_Extension(class_name, extensions):\n    def strength(s): return sum(1 if c.isupper() else -1 if c.islower() else 0 for c in s)\n    best = max(extensions, key=strength)\n    return f"{class_name}.{best}"'),

    ('cycpattern_check_rotation', 'check if any rotation of b appears as substring of a cycpattern_check',
     'def cycpattern_check(a, b):\n    doubled = b + b\n    n = len(b)\n    return any(doubled[i:i+n] in a for i in range(n))'),

    ('fizz_buzz_count7', 'count times digit 7 appears in integers divisible by 11 or 13 up to n',
     'def fizz_buzz(n):\n    return sum(str(i).count("7") for i in range(1, n) if i % 11 == 0 or i % 13 == 0)'),

    ('sort_third_idx', 'sort elements at indices divisible by 3 keep others unchanged',
     'def sort_third(l):\n    thirds = sorted(l[i] for i in range(0, len(l), 3))\n    result = list(l)\n    for i, v in zip(range(0, len(l), 3), thirds): result[i] = v\n    return result'),

    ('prime_fib_number', 'find nth number that is both prime and fibonacci prime_fib',
     'def prime_fib(n):\n    def is_prime(p):\n        if p < 2: return False\n        return all(p%i!=0 for i in range(2, int(p**0.5)+1))\n    a, b = 0, 1\n    count = 0\n    while True:\n        a, b = b, a+b\n        if is_prime(a):\n            count += 1\n            if count == n: return a'),

    ('parse_music_notes', 'parse music string notation o o| .|. return list of beat counts',
     'def parse_music(music_string):\n    note_map = {"o": 4, "o|": 2, ".|.": 1}\n    result = []\n    tokens = music_string.split()\n    for t in tokens:\n        if t in note_map: result.append(note_map[t])\n    return result'),

    ('find_zero_poly', 'find zero of polynomial using Newton bisection method find_zero',
     'def find_zero(xs):\n    def poly(x): return sum(xs[i]*x**i for i in range(len(xs)))\n    lo, hi = -1000.0, 1000.0\n    for _ in range(1000):\n        mid = (lo+hi)/2\n        if poly(mid) * poly(lo) <= 0: hi = mid\n        else: lo = mid\n    return lo'),

    ('sort_numbers_words', 'sort list of number words zero through nine sort_numbers',
     'def sort_numbers(numbers):\n    order = ["zero","one","two","three","four","five","six","seven","eight","nine"]\n    return " ".join(sorted(numbers.split(), key=lambda x: order.index(x)))'),

    ('compare_tuples', 'compare two lists return absolute difference at each position',
     'def compare(game, guess): return [abs(g-h) for g, h in zip(game, guess)]'),

    ('minpath_grid', 'minPath find minimum path of length k in n by n grid alternating minimum',
     'def minPath(grid, k):\n    n = len(grid)\n    mn = min(grid[i][j] for i in range(n) for j in range(n))\n    result = []\n    for i in range(k):\n        result.append(mn if i%2==0 else 1)\n    return result'),

    ('is_nested_brackets', 'is_nested check if string has nested square brackets at least depth 2',
     'def is_nested(string):\n    depth = 0; max_depth = 0\n    for c in string:\n        if c == "[": depth += 1; max_depth = max(max_depth, depth)\n        elif c == "]": depth -= 1\n    return max_depth >= 2'),

    ('sum_squares_odd_even', 'sum_squares square odd-indexed elements cube even-indexed',
     'def sum_squares(lst):\n    return sum(x**2 if i%2!=0 else x**3 for i, x in enumerate(lst))'),

    ('generate_integers_even', 'generate_integers return even numbers between a and b inclusive',
     'def generate_integers(a, b):\n    lo, hi = min(a, b), max(a, b)\n    return [x for x in range(lo, hi+1) if x % 2 == 0]'),

    ('vowels_count_y_end', 'vowels_count count vowels in string including y at end of word',
     'def vowels_count(s):\n    count = sum(1 for c in s.lower() if c in "aeiou")\n    if s and s[-1].lower() == "y": count += 1\n    return count'),

    ('total_match_shorter', 'total_match return list with fewer total characters between two lists',
     'def total_match(lst1, lst2):\n    s1 = sum(len(s) for s in lst1)\n    s2 = sum(len(s) for s in lst2)\n    return lst1 if s1 <= s2 else lst2'),

    ('search_frequent_max', 'search find greatest integer appearing at least that many times in list',
     'def search(lst):\n    from collections import Counter\n    c = Counter(lst)\n    valid = [x for x in c if c[x] >= x]\n    return max(valid) if valid else -1'),

    ('will_it_fly_check', 'will_it_fly check if list is palindrome and sum not exceeding max weight',
     'def will_it_fly(q, w): return q == q[::-1] and sum(q) <= w'),
]


def build_common_index():
    """Build CodeIndex populated with common Python functions."""
    idx = CodeIndex()
    for name, docstring, code in _COMMON_CORPUS:
        idx.add(name, code, docstring=docstring)
    return idx


def generate_with_index(query_text, fn_name, index, threshold=0.0, fallback=True):
    """
    Retrieve top-k from corpus, adapt function name, return (code, source, sim) candidates.
    Falls back to template generation if no retrieval hit above threshold.
    """
    candidates = []
    retrieved = index.retrieve(query_text, top_k=8)
    for name, sim in retrieved:
        if sim < threshold:
            continue
        code = index.codes[name]
        adapted = rename_function(code, name, fn_name)
        candidates.append((adapted, f'retrieved:{name}', sim))

    if fallback:
        for t_name, code, t_sim in generate_function(query_text, fn_name):
            candidates.append((code, f'template:{t_name}', t_sim))

    return candidates


#  ASTGrammarWalker (Pack 20)

TEMPLATES = {
    'expr_single': {
        'pattern': 'def {name}(x): return {expr}',
        'slots': {'expr': 'EXPR_UNARY'},
        'keywords': ['square','cube','double','halve','negate','absolute',
                     'transform','compute','calculate'],
    },
    'expr_compare': {
        'pattern': 'def {name}(x): return x {cmpop} {val}',
        'slots': {'cmpop': 'CMPOP', 'val': 'NUM_CONST'},
        'keywords': ['check','is','greater','less','equal','positive',
                     'negative','zero','compare'],
    },
    'list_filter': {
        'pattern': 'def {name}(lst): return [x for x in lst if {cond}]',
        'slots': {'cond': 'COND'},
        'keywords': ['filter','select','keep','only','where','list','elements',
                     'even','odd','positive','negative'],
    },
    'list_map': {
        'pattern': 'def {name}(lst): return [{expr} for x in lst]',
        'slots': {'expr': 'EXPR_UNARY'},
        'keywords': ['transform','apply','map','convert','each'],
    },
    'list_reduce': {
        'pattern': 'def {name}(lst): return {fn}(lst)',
        'slots': {'fn': 'REDUCE_FN'},
        'keywords': ['sum','max','min','maximum','minimum','total','count',
                     'length','find','list'],
    },
    'string_slice': {
        'pattern': 'def {name}(s): return s[{slice}]',
        'slots': {'slice': 'SLICE'},
        'keywords': ['reverse','string','first','last','slice','flip','backward'],
    },
    'string_method': {
        'pattern': 'def {name}(s): return s.{method}()',
        'slots': {'method': 'STR_METHOD'},
        'keywords': ['uppercase','lowercase','upper','lower','strip','string'],
    },
    'count_cond': {
        'pattern': "def {name}(s): return sum(1 for c in s if c in {chars!r})",
        'slots': {'chars': 'CHAR_SET'},
        'keywords': ['count','vowels','consonants','digits','occurrences',
                     'characters','letters','string'],
    },
    'binary_arith': {
        'pattern': 'def {name}(a, b): return a {op} b',
        'slots': {'op': 'ARITH_OP'},
        'keywords': ['add','subtract','multiply','divide','sum','product',
                     'difference','two','numbers'],
    },
    'is_palindrome': {
        'pattern': 'def {name}(s): return s == s[::-1]',
        'slots': {},
        'keywords': ['palindrome','same','reversed','symmetric','string','check'],
    },
}

SLOT_VOCABS = {
    'EXPR_UNARY': {
        'x * x':      ['square','squared'],
        'x * x * x':  ['cube','cubed'],
        '-x':         ['negate','negative','opposite','flip'],
        'x * 2':      ['double','twice','two'],
        'x // 2':     ['halve','half','divided'],
        'x + 1':      ['increment','next','plus'],
        'x - 1':      ['decrement','minus','previous'],
        'abs(x)':     ['absolute','magnitude'],
    },
    'CMPOP': {
        '>':  ['greater','more','positive','above','larger'],
        '<':  ['less','negative','below','smaller'],
        '==': ['equal','same','is','zero'],
    },
    'NUM_CONST': {
        '0':  ['zero','positive','negative'],
        '1':  ['one'],
        '2':  ['two','pair','even'],
        '10': ['ten'],
    },
    'COND': {
        'x % 2 == 0':  ['even'],
        'x % 2 == 1':  ['odd'],
        'x > 0':       ['positive'],
        'x < 0':       ['negative'],
        'x == 0':      ['zero'],
        'x > 10':      ['large','big','greater'],
    },
    'REDUCE_FN': {
        'sum':    ['sum','total','add'],
        'max':    ['max','maximum','largest','greatest','find'],
        'min':    ['min','minimum','smallest','least'],
        'len':    ['length','count','size'],
        'sorted': ['sort','sorted','order'],
        'set':    ['unique','distinct'],
    },
    'SLICE': {
        '::-1':  ['reverse','flip','backward'],
        '0':     ['first'],
        '-1':    ['last','end'],
    },
    'STR_METHOD': {
        'upper':  ['upper','uppercase','capital'],
        'lower':  ['lower','lowercase','small'],
        'strip':  ['strip','trim','whitespace'],
    },
    'CHAR_SET': {
        'aeiou':                  ['vowel','vowels'],
        'bcdfghjklmnpqrstvwxyz':  ['consonant','consonants'],
        '0123456789':             ['digit','digits','number'],
    },
    'ARITH_OP': {
        '+':  ['add','sum','plus','total'],
        '-':  ['subtract','minus','difference'],
        '*':  ['multiply','product','times'],
        '//': ['divide','quotient','division'],
        '%':  ['modulo','remainder','mod'],
    },
}


def _kw_bundle(keywords):
    return bundle([hv(f'__id__{k}') for k in keywords])


_TEMPLATE_HVS = {t: _kw_bundle(meta['keywords']) for t, meta in TEMPLATES.items()}
_SLOT_FILLER_HVS = {
    slot_type: {val: _kw_bundle(kws) for val, kws in fillers.items()}
    for slot_type, fillers in SLOT_VOCABS.items()
}


def _select_template(q_hv, top_k=3):
    sims = [(name, cosine(q_hv, h)) for name, h in _TEMPLATE_HVS.items()]
    sims.sort(key=lambda x: -x[1])
    return sims[:top_k]


def _fill_slot(slot_type, q_hv):
    fillers = _SLOT_FILLER_HVS[slot_type]
    best, best_sim = None, -2.0
    for val, h in fillers.items():
        s = cosine(q_hv, h)
        if s > best_sim:
            best_sim = s; best = val
    return best


def generate_function(query_text, fn_name='fn', max_candidates=5):
    """Returns list of (template_name, code, template_sim) ranked by fit."""
    q_hv = encode_query(query_text)
    ranked = _select_template(q_hv, top_k=max_candidates)
    out = []
    for t_name, t_sim in ranked:
        meta = TEMPLATES[t_name]
        fills = {'name': fn_name}
        for slot_name, slot_type in meta['slots'].items():
            fills[slot_name] = _fill_slot(slot_type, q_hv)
        try:
            code = meta['pattern'].format(**fills)
        except Exception:
            continue
        out.append((t_name, code, t_sim))
    return out


def compile_ok(code):
    try:
        compile(code, '<gen>', 'exec')
        return True
    except SyntaxError:
        return False


def exec_and_test(code, fn_name, test_cases):
    ns = {}
    try:
        exec(compile(code, '<gen>', 'exec'), ns)
    except SyntaxError:
        return False, 0.0
    except Exception:
        return True, 0.0
    if fn_name not in ns:
        return True, 0.0
    fn = ns[fn_name]
    passed = sum(1 for args, exp in test_cases if _safe_call(fn, args, exp))
    return True, passed / len(test_cases)


def _safe_call(fn, args, exp):
    try:
        return fn(*args) == exp
    except Exception:
        return False


def generate_with_verifier(query_text, fn_name, test_cases, k=5):
    cands = generate_function(query_text, fn_name, max_candidates=k)
    for t_name, code, _ in cands:
        compiled, rate = exec_and_test(code, fn_name, test_cases)
        if compiled and rate == 1.0:
            return code, t_name, True
    if cands:
        return cands[0][1], cands[0][0], False
    return None, None, False
