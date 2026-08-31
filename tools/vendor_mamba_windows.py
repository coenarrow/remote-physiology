"""Regenerate ``vendor/mamba-ssm/`` -- upstream mamba-ssm, patched to build on Windows.

Upstream ships no Windows wheel (PyPI has an sdist only; the GitHub release
assets are ``linux_x86_64`` and ``linux_aarch64``), so Windows has to compile
``selective_scan_cuda`` from source -- and that source does not compile with
MSVC. Three things stop it, all mechanical:

1. ``selective_scan_{fwd,bwd}_kernel.cuh`` put an ``#ifndef USE_ROCM`` block
   *inside* a ``BOOL_SWITCH(...)`` macro argument. A preprocessor directive in a
   macro argument is undefined behaviour; GCC processes it, MSVC (and so nvcc's
   EDG front end in Microsoft mode) rejects it with ``"#" not expected here``.
   We hoist the branch into a file-scope macro -- same code on both arms.
2. ``M_LOG2E`` needs ``_USE_MATH_DEFINES`` before ``<math.h>`` on MSVC.
3. ``BOOL_SWITCH`` passes an enclosing ``constexpr bool`` as a template argument
   from inside a lambda. MSVC's legacy lambda processor does not treat it as a
   constant expression (``error C2975``); ``/Zc:lambda`` does.

(2) and (3) are compiler flags, so they go into the vendored ``setup.py``, which
also drops the GCC spellings ``-O3 -std=c++17`` that ``cl`` merely warns about
and ignores.

The vendored tree is what ``pyproject.toml`` points ``mamba-ssm`` at on Windows,
so ``uv sync`` builds it like any other source dependency. Nothing here affects
Linux or macOS: those keep the published packages.

    uv run python tools/vendor_mamba_windows.py            # refresh the tree
    uv run python tools/vendor_mamba_windows.py --check     # CI: verify it matches

Bumping the pinned version is ``--version <new>``; re-run and commit the diff.
"""

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_VERSION = "2.3.1"
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "mamba-ssm"

# Everything the build backend needs, and nothing else: upstream's tests/ and
# egg-info are dropped so the vendored tree stays small enough to read.
KEEP = ("mamba_ssm", "csrc", "setup.py", "pyproject.toml", "setup.cfg",
        "MANIFEST.in", "README.md", "LICENSE", "AUTHORS")

SMEM_MACRO = '''
// --- rPPG-Toolbox Windows patch -------------------------------------------
// MSVC's preprocessor (which nvcc's EDG front end mimics) rejects an #if/#else
// that appears inside a macro argument, and the call site below sits inside a
// BOOL_SWITCH(...) lambda. Hoisting the branch to file scope keeps both arms
// byte-for-byte what upstream compiles, without nesting a directive.
#ifndef USE_ROCM
    #define MAMBA_SET_MAX_DYNAMIC_SMEM(kernel_, smem_) \\
        C10_CUDA_CHECK(cudaFuncSetAttribute( \\
            kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_))
#else
    #define MAMBA_SET_MAX_DYNAMIC_SMEM(kernel_, smem_) \\
        do { \\
            C10_CUDA_CHECK(cudaFuncSetAttribute( \\
                (void *) kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_)); \\
            std::cerr << "Warning: setting maxDynamicSharedMemorySize on an AMD GPU " \\
                         "is a non-op in ROCm <= 6.1. This might lead to undefined behavior." \\
                      << std::endl; \\
        } while (0)
#endif
// --- end patch -------------------------------------------------------------
'''

MSVC_FLAGS = '''
    # --- rPPG-Toolbox Windows patch ---------------------------------------
    # cl does not understand the GCC spellings above; it warns (D9002) and
    # ignores them, so state the MSVC equivalents. /Zc:lambda is required, not
    # cosmetic: without it BOOL_SWITCH's enclosing constexpr is not a constant
    # expression inside the lambda (error C2975). _USE_MATH_DEFINES is what
    # makes <math.h> declare M_LOG2E.
    if sys.platform == "win32" and not HIP_BUILD:
        extra_compile_args["cxx"] = ["/O2", "/std:c++17", "/Zc:lambda",
                                     "/D_USE_MATH_DEFINES"]
        extra_compile_args["nvcc"] = [
            flag for flag in extra_compile_args["nvcc"]
            if flag not in ("-O3", "-std=c++17")
        ] + ["-O3", "-std=c++17", "-D_USE_MATH_DEFINES",
             "-Xcompiler", "/Zc:lambda"]
    # --- end patch ---------------------------------------------------------
'''


def fetch_sdist(version):
    """Download the pinned sdist from PyPI and return (bytes, sha256)."""
    url = f"https://pypi.org/pypi/mamba-ssm/{version}/json"
    with urllib.request.urlopen(url) as response:
        release = json.load(response)
    sdists = [f for f in release["urls"] if f["packagetype"] == "sdist"]
    if not sdists:
        raise SystemExit(f"mamba-ssm {version} publishes no sdist")
    with urllib.request.urlopen(sdists[0]["url"]) as response:
        blob = response.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != sdists[0]["digests"]["sha256"]:
        raise SystemExit("sha256 mismatch on the downloaded sdist")
    return blob, digest


def patch_kernel(path):
    """Hoist the ``#ifndef USE_ROCM`` out of the BOOL_SWITCH macro argument."""
    text = path.read_text(encoding="utf-8")
    anchor = '#include "static_switch.h"\n'
    if anchor not in text:
        raise SystemExit(f"{path.name}: no static_switch.h include to anchor on")
    text = text.replace(anchor, anchor + SMEM_MACRO, 1)

    guard = "if (kSmemSize >= 48 * 1024) {"
    start = text.index(guard)
    end = text.index("}", text.index("#endif", start))
    text = text[:start] + guard + "\n" + " " * 24 + \
        "MAMBA_SET_MAX_DYNAMIC_SMEM(kernel, kSmemSize);\n" + " " * 20 + text[end:]
    path.write_text(text, encoding="utf-8")


def patch_setup(path):
    """Give MSVC flags it understands, and the two it cannot build without."""
    text = path.read_text(encoding="utf-8")
    anchor = "    ext_modules.append(\n"
    if anchor not in text:
        raise SystemExit("setup.py: no ext_modules.append to anchor on")
    return_type = text.replace(anchor, MSVC_FLAGS + "\n" + anchor, 1)
    if "import sys" not in return_type:
        raise SystemExit("setup.py: expected `import sys` to already be present")
    path.write_text(return_type, encoding="utf-8")


def build_tree(version, destination):
    """Extract, prune and patch the sdist into ``destination``."""
    blob, digest = fetch_sdist(version)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        root = f"mamba_ssm-{version}"
        members = [m for m in archive.getmembers()
                   if m.name.split("/")[1:2] and m.name.split("/")[1] in KEEP]
        archive.extractall(destination.parent, members=members, filter="data")
    extracted = destination.parent / root
    if destination.exists():
        shutil.rmtree(destination)
    extracted.rename(destination)

    kernels = destination / "csrc" / "selective_scan"
    patch_kernel(kernels / "selective_scan_bwd_kernel.cuh")
    patch_kernel(kernels / "selective_scan_fwd_kernel.cuh")
    patch_setup(destination / "setup.py")
    (destination / "VENDORED.md").write_text(
        f"""# Vendored mamba-ssm {version}

Generated by `tools/vendor_mamba_windows.py`; do not edit by hand -- re-run the
script instead so the patches stay reproducible.

- upstream: https://pypi.org/project/mamba-ssm/{version}/
- sdist sha256: `{digest}`

Used only on Windows (`pyproject.toml` maps `mamba-ssm` here under
`sys_platform == 'win32'`). Linux and macOS install the published packages.
Read the script's docstring for what the three patches are and why.
""", encoding="utf-8")
    return digest


# Building in place (uv does) drops build/ and *.egg-info/ into the tree; they
# are gitignored and are not part of what was vendored.
_BUILD_ARTEFACTS = ("build", "__pycache__")


def _is_source(relative):
    return not any(part in _BUILD_ARTEFACTS or part.endswith(".egg-info")
                   for part in relative.parts)


def tree_digest(path):
    """Order-independent hash of the vendored source, for ``--check``."""
    accumulator = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*")
                       if p.is_file() and _is_source(p.relative_to(path))):
        accumulator.update(str(file.relative_to(path)).replace("\\", "/").encode())
        accumulator.update(file.read_bytes())
    return accumulator.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--check", action="store_true",
                        help="rebuild into a temporary tree and diff, without writing")
    args = parser.parse_args()

    if args.check:
        scratch = VENDOR_DIR.parent / ".mamba-ssm-check"
        try:
            build_tree(args.version, scratch)
            if not VENDOR_DIR.exists():
                raise SystemExit(f"{VENDOR_DIR} does not exist; run without --check")
            if tree_digest(scratch) != tree_digest(VENDOR_DIR):
                raise SystemExit(f"{VENDOR_DIR} differs from a fresh vendoring of "
                                 f"mamba-ssm {args.version}; re-run without --check")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"vendor/mamba-ssm matches mamba-ssm {args.version}")
        return

    digest = build_tree(args.version, VENDOR_DIR)
    print(f"vendored mamba-ssm {args.version} -> {VENDOR_DIR}")
    print(f"sdist sha256 {digest}")


if __name__ == "__main__":
    sys.exit(main())
