# TORCWA v2 S4 External Validation

Date: 2026-04-25

## Purpose

S4 is the external RCWA/FMM reference solver for TORCWA v2 physics validation.
Default pytest does not require S4.  Instead, S4 is used to generate committed
reference fixtures under `references/s4/`, and regular tests compare TORCWA
against those fixtures whenever they are present.

Official S4 references:

- https://web.stanford.edu/group/fan/S4/
- https://web.stanford.edu/group/fan/S4/python_api.html
- https://web.stanford.edu/group/fan/S4/install.html

## Fixture Workflow

Build the official Stanford S4 Python extension from source.  On systems that
already provide Python development headers this may work directly:

```bash
git clone https://github.com/victorliu/S4.git /tmp/S4
cd /tmp/S4
make S4_pyext
export PYTHONPATH="/tmp/S4/build/lib.*:$PYTHONPATH"
```

In the current local environment, sudo is unavailable and `/usr/include` does
not contain `Python.h`.  The reproducible no-sudo path is:

```bash
bash tools/build_s4_no_sudo.sh
export PYTHONPATH="/tmp/torcwa-s4-no-sudo/S4/build/lib.linux-x86_64-cpython-312:$PYTHONPATH"
export S4_SOURCE_COMMIT="3e19a16"
export S4_BUILD_NOTES="S4 3e19a16 with local Python 3 build fixes"
```

This script downloads the matching CPython source tree only for headers,
configures it locally to generate `pyconfig.h`, checks out S4 commit `3e19a16`,
applies small Python 3.12 build fixes, builds the extension, and runs an import
smoke test.  The latest S4 `master` Python extension was also tested but imports
with `undefined symbol: Layer_Init`, which matches the upstream issue report at
https://github.com/victorliu/S4/issues/61.

Then generate fixtures:

```bash
python3 tools/generate_s4_fixtures.py --overwrite
python3 -m pytest -q
```

The generator intentionally refuses to overwrite existing fixture files unless
`--overwrite` is passed.  It also rejects the unrelated PyPI package named `S4`;
the official module must expose `S4.NewSimulation` or `S4.New`.

## Current Local Status

S4 was built without sudo by using CPython source headers and the patched
`3e19a16` S4 Python-extension commit.  The generated module exposes `S4.New`
and passes the live generator smoke test.

Current behavior:

- `python3 -m pytest -q`: passes without S4 installed or with committed
  fixtures.
- `python3 -m pytest -q -m s4_live`: passes when the no-sudo S4 build directory
  is on `PYTHONPATH`, otherwise skips.
- `tools/generate_s4_fixtures.py --overwrite`: generates all six core fixtures
  when the no-sudo S4 build directory is on `PYTHONPATH`.

## Core Gate Cases

The manifest at `references/s4/manifest.json` defines six external-reference
cases:

- `s4_uniform_slab`
- `s4_oblique_interface`
- `s4_1d_binary_grating`
- `s4_2d_rect_metasurface`
- `s4_lossy_grating`
- `s4_multilayer_stack`

Homogeneous cases use tight tolerances.  Patterned cases initially use modestly
relaxed power tolerances, including a small absolute floor, because S4 uses closed-form
pattern Fourier coefficients while TORCWA currently represents patterned
materials through sampled grids.  This is especially important for low-reflection
cases where a small absolute power difference can look large as a relative
reflection error.  The absolute floor is still tight enough to catch multilayer
coupling regressions in the patterned-stack fixture.

## Regression Caught During S4 Comparison

Comparing the updated solver against original TORCWA and S4 exposed a
multilayer-only regression introduced by the diagonal-free homogeneous
`P/Q` rewrite.  The off-diagonal signs in the expanded homogeneous block
matrices did not match the original dense assembly.  Single homogeneous slabs
and single patterned layers largely masked the issue, while the
`s4_multilayer_stack` fixture amplified it.  The S4 fixture gate now uses a
small enough patterned absolute tolerance to catch that class of regression.
