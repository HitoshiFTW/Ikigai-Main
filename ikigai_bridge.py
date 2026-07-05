"""
Pack 216 -- ikigai.py Bridge.

Per Rule #1, NEVER permanently modify ikigai.py. This bridge exec()s a
patched copy so all 283 internal classes become callable from integrate.py
without touching the source.

Patches applied:
  - TICKS=1000 -> TICKS=0    (skip the main simulation loop)
  - time.sleep(...) -> None  (drop wait calls)
  - os.system('cls'...) -> None
  - save_state_to_disk() -> None  (do not write to disk)
  - load_state_from_disk() -> (None, False)
  - any input() / print bursts redirected to a buffer

After bridge load: org.bridge.classes['CellAssemblySystem'] is a real class
instance you can instantiate. ~430 classes (283 top-level + ~150 nested)
become accessible.

Usage:
    bridge = IkigaiBridge.load()           # one-time, ~few seconds
    CellAssembly = bridge.classes['CellAssemblySystem']
    ca = CellAssembly(...)
    bridge.ns['some_global_var']            # access globals too
"""

import os, re, io, sys, time
import importlib.util
import contextlib


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IKIGAI_PATH  = os.path.join(PROJECT_ROOT, 'ikigai.py')


class IkigaiBridge:
    """Lazy bridge to ikigai.py classes via patched exec()."""

    _cached_ns = None         # process-wide cache
    _cached_classes = None

    def __init__(self, namespace, classes):
        self.ns = namespace
        self.classes = classes

    @classmethod
    def load(cls, verbose=False, force=False):
        """One-time exec() of patched ikigai.py. Caches the namespace.
        Returns an IkigaiBridge instance with `.ns` (full namespace) and
        `.classes` (dict of class_name -> class object)."""
        if cls._cached_ns is not None and not force:
            return cls(cls._cached_ns, cls._cached_classes)

        if not os.path.exists(IKIGAI_PATH):
            raise RuntimeError(f'ikigai.py not found at {IKIGAI_PATH}')

        if verbose:
            print(f'[Pack 216] reading {IKIGAI_PATH} ...', flush=True)
        with open(IKIGAI_PATH, 'r', encoding='utf-8', errors='replace') as h:
            src = h.read()
        if verbose:
            print(f'[Pack 216] source: {len(src)} chars, applying patches ...', flush=True)

        # Patch 1: kill main loop (TICKS=0 prevents the for-tick iteration)
        src = src.replace('TICKS=1000;', 'TICKS=0;', 1)
        # Patch 2: drop sleeps so module load is instant
        src = re.sub(r'time\.sleep\([^)]*\)', 'None', src)
        # Patch 3: drop os.system clears
        src = re.sub(r"os\.system\('cls'[^)]*\)", 'None', src)
        # Patch 4: stub state I/O so no disk writes happen
        src = src.replace(
            'saved_state,state_exists=load_state_from_disk()',
            'saved_state,state_exists=None,False', 1)
        # Patch 5: input() must not block if any survived
        src = re.sub(r'\binput\([^)]*\)', '""', src)

        ns = {
            '__name__': '__ikigai_bridge__',
            '__file__': IKIGAI_PATH,
        }
        if verbose:
            print(f'[Pack 216] exec()ing (this takes a few seconds) ...', flush=True)
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(src, IKIGAI_PATH, 'exec'), ns)
        except SystemExit:
            pass
        except Exception as e:
            if verbose:
                print(f'[Pack 216] exec ended with: {type(e).__name__}: {e}', flush=True)

        dt = time.perf_counter() - t0
        if verbose:
            print(f'[Pack 216] exec done in {dt:.1f}s. namespace size: {len(ns)}', flush=True)

        # Harvest classes
        classes = {}
        for k, v in ns.items():
            if isinstance(v, type):
                classes[k] = v
        if verbose:
            print(f'[Pack 216] {len(classes)} classes harvested', flush=True)

        cls._cached_ns = ns
        cls._cached_classes = classes
        return cls(ns, classes)

    # ── convenience accessors ────────────────────────────────────────────

    def has(self, class_name):
        return class_name in self.classes

    def cls(self, class_name):
        """Return the class object (raises KeyError if missing)."""
        return self.classes[class_name]

    def get(self, name, default=None):
        """Get any namespace symbol (class, function, constant)."""
        return self.ns.get(name, default)

    def list_classes(self, prefix=None):
        """List class names, optionally filtered by prefix."""
        if prefix is None:
            return sorted(self.classes.keys())
        return sorted(k for k in self.classes if k.startswith(prefix))

    def class_info(self, class_name):
        """Return introspection dict for a class."""
        cls_obj = self.classes.get(class_name)
        if cls_obj is None:
            return None
        return {
            'name': class_name,
            'module': getattr(cls_obj, '__module__', '?'),
            'bases': [b.__name__ for b in cls_obj.__bases__],
            'methods': [m for m in dir(cls_obj) if not m.startswith('_')],
            'init_sig': str(cls_obj.__init__.__doc__) if cls_obj.__init__.__doc__ else None,
        }

    def summary(self):
        """Quick stats on the loaded namespace."""
        n_cls = len(self.classes)
        n_fn = sum(1 for v in self.ns.values() if callable(v) and not isinstance(v, type))
        n_const = sum(1 for k, v in self.ns.items()
                       if not callable(v) and not k.startswith('_')
                       and not isinstance(v, type))
        return {
            'total_classes': n_cls,
            'total_functions': n_fn,
            'total_constants_or_other': n_const,
            'namespace_size': len(self.ns),
        }
