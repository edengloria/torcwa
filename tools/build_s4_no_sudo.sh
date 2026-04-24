#!/usr/bin/env bash
set -euo pipefail

S4_COMMIT="${S4_COMMIT:-3e19a16}"
BUILD_ROOT="${S4_BUILD_ROOT:-/tmp/torcwa-s4-no-sudo}"
S4_DIR="${S4_DIR:-$BUILD_ROOT/S4}"
PY_BIN="${PYTHON:-python3}"

mkdir -p "$BUILD_ROOT"

py_version="$($PY_BIN - <<'PY'
import platform
print(platform.python_version())
PY
)"
py_include="$($PY_BIN - <<'PY'
import sysconfig
print(sysconfig.get_paths().get("include", ""))
PY
)"

header_cpath=""
if [ -n "$py_include" ] && [ -f "$py_include/Python.h" ]; then
  header_cpath="$py_include"
else
  py_src="$BUILD_ROOT/Python-$py_version"
  if [ ! -f "$py_src/Include/Python.h" ] || [ ! -f "$py_src/pyconfig.h" ]; then
    archive="$BUILD_ROOT/Python-$py_version.tgz"
    curl -L -o "$archive" "https://www.python.org/ftp/python/$py_version/Python-$py_version.tgz"
    rm -rf "$py_src"
    tar -xzf "$archive" -C "$BUILD_ROOT"
    (cd "$py_src" && ./configure --prefix="$BUILD_ROOT/cpython-headers" --without-ensurepip > "$BUILD_ROOT/cpython-configure.log")
  fi
  header_cpath="$py_src/Include:$py_src"
fi

if [ ! -d "$S4_DIR/.git" ]; then
  rm -rf "$S4_DIR"
  git clone https://github.com/victorliu/S4.git "$S4_DIR"
fi

cd "$S4_DIR"
git fetch --tags origin
git checkout "$S4_COMMIT"

"$PY_BIN" - <<'PY'
from pathlib import Path

def replace(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"Could not patch {path}; expected snippet not found")
    p.write_text(text.replace(old, new))

replace("S4/S4.cpp", "if(NULL == thick || thick < 0){ ret = -3; }", "if(NULL == thick || *thick < 0){ ret = -3; }")
replace("S4/main_python.c", "char *pol;", "const char *pol;")
replace("S4/main_python.c", "data->exg[2 * i + 0] = PyInt_AsLong(pj);", "data->exg[2 * i + 0] = GetPyInt(pj);")
replace("S4/main_python.c", "if(!PyString_Check(pj))", "if(!PyUnicode_Check(pj))")
replace(
    "S4/main_python.c",
    "PyString_AsStringAndSize(pj, &pol, &polLen);",
    "pol = PyUnicode_AsUTF8AndSize(pj, &polLen);\n\t\tif(NULL == pol){ return 0; }",
)
PY

bin_dir="$BUILD_ROOT/bin"
mkdir -p "$bin_dir"
ln -sf "$(command -v "$PY_BIN")" "$bin_dir/python"

export PATH="$bin_dir:$PATH"
export CPATH="$header_cpath${CPATH:+:$CPATH}"

make objdir build/libS4.a
make S4_pyext S4_LIBNAME=build/libS4.a LIBS="-llapack -lblas"

s4_lib_dir="$(find "$S4_DIR/build" -type f -name 'S4*.so' -printf '%h\n' | head -1)"
if [ -z "$s4_lib_dir" ]; then
  echo "S4 Python extension was not produced." >&2
  exit 1
fi

PYTHONPATH="$s4_lib_dir" "$PY_BIN" - <<'PY'
import S4

assert hasattr(S4, "New")
sim = S4.New(Lattice=((1.0, 0.0), (0.0, 1.0)), NumBasis=9)
assert hasattr(sim, "GetPowerFluxByOrder")
print("S4 import smoke passed")
PY

cat <<EOF

S4 built without sudo.
Use it with:

  export PYTHONPATH="$s4_lib_dir:\$PYTHONPATH"
  export S4_SOURCE_COMMIT="$S4_COMMIT"
  export S4_BUILD_NOTES="S4 $S4_COMMIT with local Python 3 build fixes"

EOF
