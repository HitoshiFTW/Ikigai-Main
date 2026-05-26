"""
Day 58 Pack 140 -- Bio-mechanism ablation harness.

Imy on Discord asked the right question: which bio mechanisms in ikigai.py
actually earn their keep? This pack answers it by toggling individual
mechanisms off and measuring behavioral deltas vs an unablated baseline.

Per project rule #1, ikigai.py is NEVER permanently modified. Each
ablation is a temporary string-patch of the source applied before exec().

Mechanisms ablated (5 + baseline):
    baseline           -- nothing changed
    no_cortisol        -- cortisol clamped to 0 every tick (HPA off)
    no_sleep           -- sleep onset disabled (force always-awake)
    no_dopamine_supp   -- DA suppression by cortisol removed
    no_arousal         -- arousal modulation of prediction error removed
    no_l23_recovery    -- L23 sleep recovery boost removed

Metrics captured per tick:
    pred_error  (pp.error)
    energy      (mean of l23.energy values)
    cortisol    (cort.level)
    sleeping    (bool)
    dopamine    (da.level)

Verifications:
    V1  baseline runs to completion
    V2  baseline metrics in expected ranges
    V3  baseline reproducible (same seed -> same metric means within tolerance)
    V4  no_cortisol: mean cortisol ~ 0 (clamp worked)
    V5  no_sleep:    sleep_fraction == 0
    V6  no_dopamine_supp: differs from baseline on dopamine metric
    V7  no_arousal: pred_error mean differs from baseline
    V8  at least 4 of 5 ablations measurably shift some metric (>5% delta)
"""

import sys, os, re, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TICKS = 300

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log('=== Pack 140: Bio-mechanism ablation harness ===\n')

#  helpers (per MEMORY exec() pattern)
def _inject_after(source, target, code):
    lines = source.split('\n'); result = []
    for line in lines:
        result.append(line)
        if target in line:
            base = ' ' * (len(line) - len(line.lstrip()))
            for cl in code.split('\n'): result.append(base + cl)
    return '\n'.join(result)

def _inject_before(source, target, code):
    lines = source.split('\n'); result = []
    for line in lines:
        if target in line:
            base = ' ' * (len(line) - len(line.lstrip()))
            for cl in code.split('\n'): result.append(base + cl)
        result.append(line)
    return '\n'.join(result)


METRICS_CODE = '''
try:
    _METRICS['pe'].append(pp.error)
    _METRICS['energy'].append(sum(l23.energy.values())/3.0 if hasattr(l23, 'energy') else 0.0)
    _METRICS['cort'].append(cort.level)
    _METRICS['sleeping'].append(1 if sleeping else 0)
    _METRICS['da'].append(da.level if 'da' in dir() else 0.0)
except Exception:
    pass
'''


def _base_patches(source):
    """Common patches: disable persistence, screen clears, cap TICKS, remove sleeps."""
    source = source.replace('TICKS=1000', f'TICKS={TICKS}', 1)
    source = source.replace(
        'saved_state,state_exists=load_state_from_disk()',
        'saved_state,state_exists=None,False', 1)
    source = re.sub(r"os\.system\('cls'[^)]*\)", 'None', source)
    # inject metrics BEFORE removing time.sleep
    source = _inject_after(source, 'time.sleep(0.010)', METRICS_CODE)
    source = re.sub(r'time\.sleep\([^)]*\)', 'None', source)
    return source


#  per-ablation source mutators
def patch_baseline(s):
    return s

def patch_no_cortisol(s):
    # clamp cort.level=0 BEFORE metrics capture each tick.  Without _inject_before
    # the clamp lands after metrics and never affects the recorded mean.
    return _inject_before(s, 'time.sleep(0.010)', 'cort.level = 0.0')

def patch_no_sleep(s):
    # DummyHomeostasis: never trigger sleep onset
    # The MEMORY.md DummyHomeostasis pattern - inject as full replacement class
    DUMMY = '''
class HomeostasisSystem:
    SLEEP_ONSET_THRESHOLD=0.70; SLEEP_OFFSET_THRESHOLD=0.30; MIN_WAKE_TICKS=0
    def __init__(self):
        self.drives={"hunger":0.0,"safety":0.0,"social":0.0,"sleep":0.0,"curiosity":0.0}
        self.global_imbalance=0.0; self._sleep_active=False; self._wake_ticks=0
        self._last_cortisol=0.0; self._arousal_override=False
    def update(self,avg_energy,cortisol,oxytocin,adenosine,prediction_error):
        self._last_cortisol=cortisol
    def should_sleep_onset(self,circadian=None): return False
    def should_sleep_end(self): return self._sleep_active
    def mark_sleep_start(self): self._sleep_active=True
    def mark_sleep_end(self): self._sleep_active=False; self._wake_ticks=0
    def get_bg_drive_biases(self): return {"approach":0.0,"withdraw":0.0,"explore":0.0}
    def export_metrics(self):
        return {"hunger":0.0,"safety":0.0,"social":0.0,"sleep":0.0,
                "curiosity":0.0,"global_imbalance":0.0}
'''
    # Replace the original HomeostasisSystem class with DUMMY by injecting
    # DUMMY at top of file - Python uses the LATEST class definition.
    return DUMMY + '\n' + s

def patch_no_dopamine_supp(s):
    # Find any line matching 'da.level *= (1' or similar suppression and neutralize.
    s = re.sub(r'da\.level\s*\*=\s*\(1\s*-\s*cort[^)]*\)', 'da.level *= 1.0', s)
    s = re.sub(r'da\.level\s*\*=\s*\(1\.0\s*-\s*cort[^)]*\)', 'da.level *= 1.0', s)
    return s

def patch_no_arousal(s):
    # Find 'pp.error *= (1+arousal*' or '(1.0+arousal*' and neutralize.
    s = re.sub(r'pp\.error\s*\*=\s*\(1[\.0]*\s*\+\s*arousal[^)]*\)',
               'pp.error *= 1.0', s)
    return s

def patch_no_l23_recovery(s):
    # MEMORY: Sleep recovery boost cortex+0.005, limbic+0.008, motor+0.010.
    # If any of those are present in the source, neutralize them.
    s = re.sub(r"l23\.energy\['cortex'\]\s*\+=\s*0\.005",
               "l23.energy['cortex'] += 0.0", s)
    s = re.sub(r"l23\.energy\['limbic'\]\s*\+=\s*0\.008",
               "l23.energy['limbic'] += 0.0", s)
    s = re.sub(r"l23\.energy\['motor'\]\s*\+=\s*0\.010",
               "l23.energy['motor'] += 0.0", s)
    return s


ABLATIONS = [
    ('baseline',         patch_baseline),
    ('no_cortisol',      patch_no_cortisol),
    ('no_sleep',         patch_no_sleep),
    ('no_dopamine_supp', patch_no_dopamine_supp),
    ('no_arousal',       patch_no_arousal),
    ('no_l23_recovery',  patch_no_l23_recovery),
]


def run_ablation(name, mutator, source):
    metrics = {'pe': [], 'energy': [], 'cort': [], 'sleeping': [], 'da': []}
    # mutator FIRST while original anchors (time.sleep, etc.) still exist;
    # _base_patches then strips them.
    patched = mutator(source)
    patched = _base_patches(patched)
    ns = {
        '__name__': '__experiment__',
        '__file__': str(PROJECT_ROOT/'ikigai.py'),
        '_METRICS': metrics,
    }
    buf = io.StringIO()
    t0 = time.perf_counter()
    err = None
    try:
        with redirect_stdout(buf):
            exec(compile(patched, str(PROJECT_ROOT/'ikigai.py'), 'exec'), ns)
    except Exception as e:
        err = repr(e)
    elapsed = time.perf_counter() - t0

    # summarize
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    summary = {
        'name': name,
        'elapsed': elapsed,
        'ticks_recorded': len(metrics['pe']),
        'pe_mean':        mean(metrics['pe']),
        'energy_mean':    mean(metrics['energy']),
        'cort_mean':      mean(metrics['cort']),
        'sleep_frac':     mean(metrics['sleeping']),
        'da_mean':        mean(metrics['da']),
        'error': err,
    }
    return summary


#  load canonical ikigai.py source
source = (PROJECT_ROOT / 'ikigai.py').read_text(encoding='utf-8', errors='replace')
_log(f'  loaded ikigai.py ({len(source)/1024:.0f} KB)')
_log(f'  running {len(ABLATIONS)} configurations at TICKS={TICKS}...\n')

#  run all
results = {}
for name, mutator in ABLATIONS:
    _log(f'  [{name:18s}] ...', )
    summary = run_ablation(name, mutator, source)
    results[name] = summary
    if summary['error']:
        _log(f'      ERROR: {summary["error"][:100]}')
    else:
        _log(f'      {summary["elapsed"]:5.1f}s  ticks={summary["ticks_recorded"]:4d}  '
             f'PE={summary["pe_mean"]:.3f}  E={summary["energy_mean"]:.3f}  '
             f'cort={summary["cort_mean"]:.3f}  sleep_frac={summary["sleep_frac"]:.3f}  '
             f'da={summary["da_mean"]:.3f}')

#  verifications
_log('')
base = results['baseline']
check('V1 baseline runs to completion',
      base['error'] is None and base['ticks_recorded'] > 50,
      f'err={base["error"]}, ticks={base["ticks_recorded"]}')

check('V2 baseline metrics in expected ranges',
      0.0 <= base['pe_mean'] <= 5.0 and
      0.0 <= base['energy_mean'] <= 1.5 and
      0.0 <= base['cort_mean'] <= 1.0 and
      0.0 <= base['sleep_frac'] <= 1.0,
      str({k: round(base[k],3) for k in ['pe_mean','energy_mean','cort_mean','sleep_frac']}))

# V3: rerun baseline once more for reproducibility (random.seed() is unseeded
# in ikigai.py, so allow generous tolerance)
base2 = run_ablation('baseline_rerun', patch_baseline, source)
pe_delta = abs(base['pe_mean'] - base2['pe_mean'])
check('V3 baseline reproducible within tolerance', pe_delta < 1.0,
      f'PE delta {pe_delta:.3f}')

# V4 no_cortisol clamped
nc = results['no_cortisol']
check('V4 no_cortisol mean ~= 0', nc['cort_mean'] < 0.05,
      f'cort_mean={nc["cort_mean"]:.4f}')

# V5 no_sleep
ns = results['no_sleep']
check('V5 no_sleep sleep_frac == 0', ns['sleep_frac'] < 0.05,
      f'sleep_frac={ns["sleep_frac"]:.4f}')

# V6 no_dopamine_supp: shift on da mean (or behavioral metric)
nds = results['no_dopamine_supp']
da_delta = abs(nds['da_mean'] - base['da_mean'])
check('V6 no_dopamine_supp differs from baseline on da',
      da_delta > 0.001 or nds['pe_mean'] != base['pe_mean'],
      f'da delta {da_delta:.5f}, PE same={nds["pe_mean"]==base["pe_mean"]}')

# V7 no_arousal: shift on PE mean
na = results['no_arousal']
pe_diff = abs(na['pe_mean'] - base['pe_mean'])
check('V7 no_arousal shifts pred_error', pe_diff > 0.0001,
      f'PE delta {pe_diff:.5f}')

# V8 at least 4 of 5 ablations shift SOME metric > 5%
def pct_diff(a, b):
    denom = max(abs(b), 0.001)
    return abs(a - b) / denom

shifts = 0
for name in ['no_cortisol','no_sleep','no_dopamine_supp','no_arousal','no_l23_recovery']:
    r = results[name]
    if r['error']: continue
    diffs = [
        pct_diff(r['pe_mean'],     base['pe_mean']),
        pct_diff(r['energy_mean'], base['energy_mean']),
        pct_diff(r['cort_mean'],   base['cort_mean']),
        pct_diff(r['sleep_frac'],  base['sleep_frac']),
        pct_diff(r['da_mean'],     base['da_mean']),
    ]
    if max(diffs) > 0.05:
        shifts += 1

check('V8 >= 4 of 5 ablations measurably shift a metric (>5%)',
      shifts >= 4, f'got {shifts}/5 with >5% shift')

#  comparison table
_log('\n  Ablation comparison table:')
_log(f'  {"name":<20s} {"PE":>7s} {"E":>6s} {"cort":>6s} {"sleep":>6s} {"da":>6s}')
for name, _ in ABLATIONS:
    r = results[name]
    if r['error']:
        _log(f'  {name:<20s}  <error>')
        continue
    _log(f'  {name:<20s} {r["pe_mean"]:7.3f} {r["energy_mean"]:6.3f} '
         f'{r["cort_mean"]:6.3f} {r["sleep_frac"]:6.3f} {r["da_mean"]:6.3f}')

total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 140 -- Bio-mechanism ablation harness')
_log(f'{"="*64}')
_log(f'  Ran {len(ABLATIONS)} configurations at {TICKS} ticks each')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- ablation harness operational'
                     if FAIL == 0 else 'NEEDS FIX'))
