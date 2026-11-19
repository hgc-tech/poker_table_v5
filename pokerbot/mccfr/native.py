"""Build and load the C++ engine (native/).

`build()` compiles two targets with the system C++ compiler:

* `native/pbcfr` (`pbcfr.exe` on Windows): builds the card abstraction,
  estimates memory and trains blueprints.
* `native/libpbcfr5.so` (`.dylib` on macOS): the run-time library the bots
  call through ctypes for blueprint lookups and search.

`Engine` wraps the library. `build_v4_library` compiles the run-time library
of a v4 install, so v5 can play matches against v4 in one process.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
SOURCES = ["pb_core.h", "pb_abstraction.h", "pb_game.h", "pb_train.h", "main.cpp", "pb_runtime.cpp"]

_SYSTEM = platform.system()
EXE = NATIVE / ("pbcfr.exe" if _SYSTEM == "Windows" else "pbcfr")
LIB = NATIVE / {"Windows": "pbcfr5.dll", "Darwin": "libpbcfr5.dylib"}.get(_SYSTEM, "libpbcfr5.so")


def _stale(target: Path) -> bool:
    if not target.exists():
        return True
    t = target.stat().st_mtime
    return any((NATIVE / s).stat().st_mtime > t for s in SOURCES)


def _compiler() -> list[str]:
    for env in ("CXX",):
        if os.environ.get(env):
            return [os.environ[env]]
    for c in ("g++", "clang++", "c++"):
        if shutil.which(c):
            return [c]
    if shutil.which("cl"):
        return ["cl"]
    raise RuntimeError("No C++ compiler found. Install g++ or clang++ (Linux, macOS) or "
                       "Visual Studio Build Tools (Windows), or build with CMake (see native/CMakeLists.txt).")


# Apple clang on Apple silicon rejects -march=native; its default target is already the native CPU.
_ARCH = [] if (_SYSTEM == "Darwin" and platform.machine() == "arm64") else ["-march=native"]


def build(force: bool = False, quiet: bool = False) -> None:
    """Compile the trainer and the run-time library if a source changed."""
    cxx = _compiler()
    msvc = cxx[0].endswith("cl") or cxx[0].endswith("cl.exe")
    jobs = []
    if force or _stale(EXE):
        if msvc:
            jobs.append(cxx + ["/O2", "/std:c++17", "/EHsc", "main.cpp", f"/Fe:{EXE.name}"])
        else:
            jobs.append(cxx + ["-O3", *_ARCH, "-std=c++17", "-pthread", "-Wno-unused-result",
                               "main.cpp", "-o", EXE.name])
    if force or _stale(LIB):
        if msvc:
            jobs.append(cxx + ["/O2", "/std:c++17", "/EHsc", "/LD", "pb_runtime.cpp", f"/Fe:{LIB.name}"])
        else:
            flags = ["-O3", *_ARCH, "-std=c++17", "-pthread", "-fPIC", "-fvisibility=hidden",
                     "-Wno-unused-result"]
            flags += ["-dynamiclib"] if _SYSTEM == "Darwin" else ["-shared"]
            jobs.append(cxx + flags + ["pb_runtime.cpp", "-o", LIB.name])
    for cmd in jobs:
        if not quiet:
            print("Compiling:", " ".join(cmd), file=sys.stderr)
        r = subprocess.run(cmd, cwd=NATIVE, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{r.stdout}\n{r.stderr}")


def build_v4_library(v4_root: str | Path) -> Path:
    """Compile <v4_root>/native/pb_runtime.cpp into native/build/ (once) and return the library path."""
    src = Path(v4_root).expanduser().resolve() / "native" / "pb_runtime.cpp"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found: point --v4 at the v4 poker_table folder")
    out_dir = NATIVE / "build"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / ("libpbcfr_v4.dylib" if _SYSTEM == "Darwin" else "libpbcfr_v4.so")
    if out.exists() and out.stat().st_mtime >= max(p.stat().st_mtime for p in src.parent.glob("*.*")):
        return out
    cxx = _compiler()
    flags = ["-O3", *_ARCH, "-std=c++17", "-pthread", "-fPIC", "-fvisibility=hidden", "-Wno-unused-result"]
    flags += ["-dynamiclib"] if _SYSTEM == "Darwin" else ["-shared"]
    r = subprocess.run(cxx + flags + [str(src), "-o", str(out)], cwd=src.parent, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Compiling the v4 library failed:\n{r.stdout}\n{r.stderr}")
    return out


class EngineError(RuntimeError):
    pass


class Engine:
    """The run-time library: card abstraction plus loaded blueprints."""

    def __init__(self, abstraction_dir: str | Path, blueprint_dir: str | Path | None = None,
                 lib: str | Path | None = None):
        if lib is None:
            build(quiet=True)
        self.lib = ctypes.CDLL(str(lib or LIB))
        L = self.lib
        L.pb_open.restype = ctypes.c_void_p
        L.pb_open.argtypes = [ctypes.c_char_p]
        L.pb_load.restype = ctypes.c_int
        L.pb_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        for f in (L.pb_decide, L.pb_contexts):
            f.restype = ctypes.c_char_p
            f.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.pb_info.restype = ctypes.c_char_p
        L.pb_info.argtypes = [ctypes.c_void_p]
        L.pb_error.restype = ctypes.c_char_p
        L.pb_close.argtypes = [ctypes.c_void_p]
        self.handle = L.pb_open(str(abstraction_dir).encode())
        if not self.handle:
            raise EngineError(L.pb_error().decode())
        if blueprint_dir is not None:
            for p in sorted(Path(blueprint_dir).glob("*.strat")):
                self.load(p)

    def load(self, path: str | Path) -> int:
        i = self.lib.pb_load(self.handle, str(path).encode())
        if i < 0:
            raise EngineError(self.lib.pb_error().decode())
        return i

    def info(self) -> dict:
        return json.loads(self.lib.pb_info(self.handle).decode())

    def _call(self, fn, request: str):
        out = json.loads(fn(self.handle, request.encode()).decode())
        if isinstance(out, dict) and "error" in out:
            raise EngineError(out["error"])
        return out

    def decide(self, request: str) -> dict:
        return self._call(self.lib.pb_decide, request)

    def contexts(self, request: str) -> list:
        return self._call(self.lib.pb_contexts, request)

    def close(self):
        if self.handle:
            self.lib.pb_close(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
