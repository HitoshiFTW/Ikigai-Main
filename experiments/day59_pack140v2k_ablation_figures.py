"""
Day 59 Pack 140v -- Ablation harness with figures.

Same harness as Pack 140 (Day 58) but captures full per-tick time-series
and renders a single-page PDF with all ablations on one canvas.  Built
for sharing on Discord -- visual answer to Imy's "which mechanisms earn
their keep" critique.

Output: experiments/_pack140v_ablation_results.pdf  (single page)

Ablations (5 + baseline):
    baseline           -- nothing changed
    no_cortisol        -- cortisol clamped to 0 every tick (HPA off)
    no_sleep           -- sleep onset disabled (force always-awake)
    no_dopamine_supp   -- DA suppression by cortisol removed
    no_arousal         -- arousal modulation of prediction error removed
    no_l23_recovery    -- L23 sleep recovery boost removed

Metrics captured per tick:
    pred_error, energy (mean of cortex/limbic/motor), cortisol, sleeping (0/1),
    dopamine.

Verifications:
    V1  PDF generated at known path
    V2  baseline runs (>=500 ticks recorded)
    V3  all 6 configurations complete without unhandled errors
    V4  no_cortisol mean cort ~= 0
    V5  no_sleep sleep_frac == 0
    V6  at least 4 of 5 ablations measurably shift some metric (>5%)
    V7  PDF file > 5 KB on disk (not corrupt-empty)
    V8  comparison CSV saved alongside
"""

import sys, os, re, io, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless render
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TICKS = 2000   # extended so adenosine ramp triggers sleep onset
OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = OUT_DIR / '_pack140v2k_ablation_results.pdf'
CSV_PATH = OUT_DIR / '_pack140v2k_ablation_results.csv'

PASS = 0; FAIL = 0
def check(name, cond, detail=''):
    global PASS, FAIL
    if cond: print(f'  [PASS] {name}', flush=True); PASS += 1
    else:    print(f'  [FAIL] {name}  {detail}', flush=True); FAIL += 1
def _log(s): print(s, flush=True)

_log(f'=== Pack 140v: Ablation harness with figures (TICKS={TICKS}) ===\n')

# ---- exec() patch helpers (per memory pattern) ----
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
    _METRICS['energy'].append(sum(l23.energy.values())/3.0 if hasattr(l23,'energy') else 0.0)
    _METRICS['cort'].append(cort.level)
    _METRICS['sleeping'].append(1 if sleeping else 0)
    _METRICS['da'].append(da.level if 'da' in dir() else 0.0)
except Exception:
    pass
'''

def _base_patches(source):
    source = source.replace('TICKS=1000', f'TICKS={TICKS}', 1)
    source = source.replace(
        'saved_state,state_exists=load_state_from_disk()',
        'saved_state,state_exists=None,False', 1)
    source = re.sub(r"os\.system\('cls'[^)]*\)", 'None', source)
    source = _inject_after(source, 'time.sleep(0.010)', METRICS_CODE)
    source = re.sub(r'time\.sleep\([^)]*\)', 'None', source)
    return source

# ---- ablation mutators (each receives raw source) ----
def patch_baseline(s):       return s

def patch_no_cortisol(s):
    return _inject_before(s, 'time.sleep(0.010)', 'cort.level = 0.0')

def patch_no_sleep(s):
    # Monkey-patch the instance right after it's constructed; Python class
    # ordering breaks the DummyHomeostasis-prepend approach.  Override the
    # method on the instance instead.
    return _inject_after(s,
        'homeostasis = HomeostasisSystem()',
        'homeostasis.should_sleep_onset = lambda *a, **kw: False')

def patch_no_dopamine_supp(s):
    s = re.sub(r'da\.level\s*\*=\s*\(1\s*-\s*cort[^)]*\)', 'da.level *= 1.0', s)
    s = re.sub(r'da\.level\s*\*=\s*\(1\.0\s*-\s*cort[^)]*\)', 'da.level *= 1.0', s)
    return s

def patch_no_arousal(s):
    s = re.sub(r'pp\.error\s*\*=\s*\(1[\.0]*\s*\+\s*arousal[^)]*\)', 'pp.error *= 1.0', s)
    return s

def patch_no_l23_recovery(s):
    s = re.sub(r"l23\.energy\['cortex'\]\s*\+=\s*0\.005", "l23.energy['cortex'] += 0.0", s)
    s = re.sub(r"l23\.energy\['limbic'\]\s*\+=\s*0\.008", "l23.energy['limbic'] += 0.0", s)
    s = re.sub(r"l23\.energy\['motor'\]\s*\+=\s*0\.010", "l23.energy['motor'] += 0.0", s)
    return s

ABLATIONS = [
    ('baseline',         patch_baseline),
    ('no_cortisol',      patch_no_cortisol),
    ('no_sleep',         patch_no_sleep),
    ('no_dopamine_supp', patch_no_dopamine_supp),
    ('no_arousal',       patch_no_arousal),
    ('no_l23_recovery',  patch_no_l23_recovery),
]

def run_one(name, mutator, source):
    metrics = {'pe': [], 'energy': [], 'cort': [], 'sleeping': [], 'da': []}
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
    return {'name': name, 'elapsed': time.perf_counter()-t0, 'metrics': metrics, 'error': err}

# ---- load + run ----
source = (PROJECT_ROOT / 'ikigai.py').read_text(encoding='utf-8', errors='replace')
_log(f'  loaded ikigai.py ({len(source)/1024:.0f} KB)\n')

results = {}
for name, mutator in ABLATIONS:
    _log(f'  [{name:18s}] running {TICKS} ticks ...')
    r = run_one(name, mutator, source)
    results[name] = r
    m = r['metrics']
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    if r['error']:
        _log(f'      ERROR: {r["error"][:100]}')
    else:
        _log(f'      {r["elapsed"]:5.1f}s  ticks={len(m["pe"]):4d}  '
             f'PE={mean(m["pe"]):.3f}  E={mean(m["energy"]):.3f}  '
             f'cort={mean(m["cort"]):.3f}  sleep={mean(m["sleeping"]):.3f}  '
             f'da={mean(m["da"]):.3f}')

# ---- verifications ----
_log('')
base = results['baseline']
def base_mean(k): return sum(base['metrics'][k])/len(base['metrics'][k]) if base['metrics'][k] else 0.0
check('V2 baseline runs (>=500 ticks recorded)',
      base['error'] is None and len(base['metrics']['pe']) >= 500)

n_complete = sum(1 for r in results.values() if r['error'] is None)
check('V3 all configurations complete', n_complete == len(ABLATIONS),
      f'{n_complete}/{len(ABLATIONS)} clean')

nc_cort = sum(results['no_cortisol']['metrics']['cort']) / max(1, len(results['no_cortisol']['metrics']['cort']))
check('V4 no_cortisol clamped (mean cort ~ 0)', nc_cort < 0.05,
      f'mean={nc_cort:.4f}')

ns_sleep = sum(results['no_sleep']['metrics']['sleeping']) / max(1, len(results['no_sleep']['metrics']['sleeping']))
check('V5 no_sleep sleep_frac == 0', ns_sleep < 0.01,
      f'sleep_frac={ns_sleep:.4f}')

def pct_diff(a, b): return abs(a-b)/max(abs(b),0.001)
shifts = 0
for name in ['no_cortisol','no_sleep','no_dopamine_supp','no_arousal','no_l23_recovery']:
    r = results[name]; m = r['metrics']
    if r['error']: continue
    diffs = [
        pct_diff(sum(m['pe'])/max(1,len(m['pe'])), base_mean('pe')),
        pct_diff(sum(m['energy'])/max(1,len(m['energy'])), base_mean('energy')),
        pct_diff(sum(m['cort'])/max(1,len(m['cort'])), base_mean('cort')),
        pct_diff(sum(m['sleeping'])/max(1,len(m['sleeping'])), base_mean('sleeping')),
        pct_diff(sum(m['da'])/max(1,len(m['da'])), base_mean('da')),
    ]
    if max(diffs) > 0.05: shifts += 1
check('V6 >= 4 of 5 ablations measurably shift a metric (>5%)', shifts >= 4,
      f'{shifts}/5')

# ---- write CSV ----
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ablation','ticks','pe_mean','energy_mean','cort_mean','sleep_frac','da_mean'])
    for name in [n for n,_ in ABLATIONS]:
        r = results[name]; m = r['metrics']
        def mm(k): return sum(m[k])/len(m[k]) if m[k] else 0.0
        w.writerow([name, len(m['pe']), f'{mm("pe"):.5f}', f'{mm("energy"):.5f}',
                    f'{mm("cort"):.5f}', f'{mm("sleeping"):.5f}', f'{mm("da"):.5f}'])
check('V8 CSV saved', CSV_PATH.exists())

# ---- render figures into ONE PDF page ----
_log(f'\n  rendering figures -> {PDF_PATH}')

COLORS = {
    'baseline':         '#222222',
    'no_cortisol':      '#d62728',
    'no_sleep':         '#1f77b4',
    'no_dopamine_supp': '#ff7f0e',
    'no_arousal':       '#9467bd',
    'no_l23_recovery':  '#2ca02c',
}

# Build a single 8.5x11 figure with 6 subplots: 2 rows x 3 cols
fig = plt.figure(figsize=(11.0, 8.5))
fig.suptitle('Ikigai bio-mechanism ablation -- which mechanisms earn their keep',
             fontsize=14, fontweight='bold')
fig.text(0.5, 0.945,
         f'{TICKS} ticks per configuration. ikigai.py is never permanently modified; each ablation is a temporary exec() patch.',
         ha='center', fontsize=9, style='italic', color='#555')

# panel A: mean-of-metric bar chart per ablation (5 metrics)
metric_keys = [('pe','PE'),('energy','Energy'),('cort','Cortisol'),('sleeping','Sleep frac'),('da','Dopamine')]
ax = fig.add_subplot(2, 3, 1)
import numpy as np
ablation_names = [n for n,_ in ABLATIONS]
bar_w = 0.13
x = np.arange(len(metric_keys))
for i, name in enumerate(ablation_names):
    r = results[name]; m = r['metrics']
    means = [sum(m[k])/max(1,len(m[k])) for k,_ in metric_keys]
    ax.bar(x + (i - len(ablation_names)/2) * bar_w, means, bar_w,
           label=name, color=COLORS[name], edgecolor='black', linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels([lbl for _,lbl in metric_keys], fontsize=8)
ax.set_title('A. Metric means by ablation', fontsize=10)
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.grid(axis='y', linestyle=':', alpha=0.4)

# panel B/C/D: time-series of cortisol, dopamine, energy
def add_timeseries(ax, key, label):
    for name in ablation_names:
        m = results[name]['metrics']
        if not m[key]: continue
        ax.plot(m[key], color=COLORS[name], label=name, lw=1.0, alpha=0.85)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('tick', fontsize=8)
    ax.grid(linestyle=':', alpha=0.4)
    ax.tick_params(labelsize=8)

add_timeseries(fig.add_subplot(2,3,2), 'cort',     'B. Cortisol over time')
add_timeseries(fig.add_subplot(2,3,3), 'da',       'C. Dopamine over time')
add_timeseries(fig.add_subplot(2,3,4), 'energy',   'D. Mean L23 energy')
add_timeseries(fig.add_subplot(2,3,5), 'pe',       'E. Prediction error')

# panel F: delta-vs-baseline table as text
ax = fig.add_subplot(2,3,6); ax.axis('off')
ax.set_title('F. Delta vs baseline (%)', fontsize=10)
def mm(r, k):
    return sum(r['metrics'][k])/max(1,len(r['metrics'][k]))
header = f'{"ablation":<18s}{"PE":>8s}{"E":>8s}{"cort":>8s}{"sleep":>8s}{"da":>8s}'
lines = [header, '-'*58]
for name in ablation_names:
    r = results[name]
    if r['error']:
        lines.append(f'{name:<18s}  <error>'); continue
    if name == 'baseline':
        lines.append(f'{name:<18s}{mm(r,"pe"):>8.3f}{mm(r,"energy"):>8.3f}'
                     f'{mm(r,"cort"):>8.3f}{mm(r,"sleeping"):>8.3f}{mm(r,"da"):>8.3f}')
    else:
        def pct(a, b):
            if abs(b) < 1e-6: return 0.0
            return 100.0 * (a - b) / abs(b)
        lines.append(f'{name:<18s}{pct(mm(r,"pe"),mm(base,"pe")):>+7.1f}%'
                     f'{pct(mm(r,"energy"),mm(base,"energy")):>+7.1f}%'
                     f'{pct(mm(r,"cort"),mm(base,"cort")):>+7.1f}%'
                     f'{pct(mm(r,"sleeping"),mm(base,"sleeping")):>+7.1f}%'
                     f'{pct(mm(r,"da"),mm(base,"da")):>+7.1f}%')
ax.text(0.0, 0.95, '\n'.join(lines), ha='left', va='top',
        family='monospace', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])
with PdfPages(str(PDF_PATH)) as pdf:
    pdf.savefig(fig)
plt.close(fig)

pdf_size = PDF_PATH.stat().st_size if PDF_PATH.exists() else 0
_log(f'  PDF saved, size = {pdf_size/1024:.1f} KB')
check('V1 PDF generated at known path', PDF_PATH.exists())
check('V7 PDF file > 5 KB on disk', pdf_size > 5_000,
      f'{pdf_size} bytes')

# ---- summary ----
total = PASS + FAIL
_log(f'\n{"="*64}')
_log(f'Pack 140v -- Ablation harness with figures')
_log(f'{"="*64}')
_log(f'  PDF: {PDF_PATH}')
_log(f'  CSV: {CSV_PATH}')
_log(f'  {PASS}/{total} PASS  ({FAIL} FAIL)')
_log('  STATUS: ' + ('SHIP -- figures ready for Discord'
                     if FAIL == 0 else 'NEEDS FIX'))
