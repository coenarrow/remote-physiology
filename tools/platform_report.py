# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Collect the platform facts that gate the Phase 3.5 dependency refresh.

Run this on every machine the project targets (Windows dev box, Linux HPC,
macOS) and commit the report it writes to ``docs/plans/platform-reports/``.
The reports together define the constraint set — CUDA driver ceilings, MSVC
and gcc versions, HPC module availability, torch wheel arch lists — from
which the upgrade targets are chosen (see the roadmap, Phase 3.5).

Deliberately **stdlib-only** and safe for an HPC login node: it reads
metadata and runs version commands, nothing computational. The same command
works on every machine:

    uv run tools/platform_report.py

The inline script metadata above makes ``uv run`` treat this as a
self-contained script — it does NOT sync the project environment, so on a
fresh clone (HPC login node) nothing heavy is installed, and uv supplies a
Python even where none is on PATH.

If a project ``.venv`` exists it is additionally introspected (torch build,
GPU visibility, every installed distribution) via its own interpreter; where
none exists the report says so and ``uv.lock`` remains the reference.

On a SLURM cluster the login node has no GPU, so ``nvidia-smi`` will fail
there; the modules + sinfo sections still capture what matters. To also
record driver/GPU facts, re-run this script inside a brief ``salloc`` on a
GPU partition — the compute node's hostname gives it a separate report file.
"""

import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "docs" / "plans" / "platform-reports"

# What the venv's own interpreter is asked to print (kept inline so the
# script stays a single stdlib-only file).
VENV_PROBE = r"""
import sys
print("python:", sys.version.replace("\n", " "))
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    try:
        print("torch.cuda.get_arch_list():", torch.cuda.get_arch_list())
    except Exception as err:
        print("arch list unavailable:", err)
    if torch.cuda.is_available():
        try:
            print("cudnn:", torch.backends.cudnn.version())
        except Exception:
            pass
        for i in range(torch.cuda.device_count()):
            print("gpu", i, torch.cuda.get_device_name(i),
                  "capability", torch.cuda.get_device_capability(i))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        print("mps available:", mps.is_available())
except Exception as err:
    print("torch introspection failed:", repr(err))
from importlib import metadata
dists = []
for dist in metadata.distributions():
    name = dist.metadata["Name"] if dist.metadata else None
    if name:
        dists.append((name.lower(), name, dist.version))
dists.sort()
print("--- installed distributions (%d) ---" % len(dists))
for _key, name, version in dists:
    print("%s==%s" % (name, version))
"""


def run(cmd, shell=False, timeout=60):
    """Run a command, returning (ok, combined stdout+stderr text)."""
    try:
        proc = subprocess.run(
            cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, universal_newlines=True,
        )
    except FileNotFoundError:
        return False, "command not found"
    except subprocess.TimeoutExpired:
        return False, "timed out after %ss" % timeout
    except OSError as err:
        return False, "failed to run: %s" % err
    text = (proc.stdout or "").strip() or "(no output)"
    if proc.returncode != 0:
        text += "\n[exit code %d]" % proc.returncode
    return proc.returncode == 0, text


class Report:
    def __init__(self):
        self.lines = []

    def heading(self, title):
        self.lines += ["", "## " + title, ""]

    def note(self, text):
        self.lines += [text, ""]

    def command(self, label, cmd, shell=False, timeout=60, shown=None):
        _ok, output = run(cmd, shell=shell, timeout=timeout)
        if shown is None:
            shown = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        self.lines += ["**`%s`**%s" % (shown, " — " + label if label else ""),
                       "", "```text", output, "```", ""]

    def text(self):
        return "\n".join(self.lines).rstrip() + "\n"


def os_key():
    return {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(
        platform.system(), platform.system().lower() or "unknown")


def section_system(report):
    report.heading("System")
    report.note("- platform: `%s`\n- machine: `%s`\n- hostname: `%s`" % (
        platform.platform(), platform.machine(), platform.node()))
    system = platform.system()
    if system == "Linux":
        report.command("distro", ["cat", "/etc/os-release"])
        report.command("glibc floor for manylinux wheels", "ldd --version | head -1", shell=True)
    elif system == "Darwin":
        report.command("macOS version", ["sw_vers"])
        report.command("chip", ["sysctl", "-n", "machdep.cpu.brand_string"])
    for var in ("CUDA_HOME", "CUDA_PATH", "CC", "CXX"):
        value = os.environ.get(var)
        if value:
            report.note("- env `%s` = `%s`" % (var, value))


def section_python(report):
    report.heading("Python and uv")
    report.note("script interpreter: `%s`\n\n```text\n%s\n```" % (
        sys.executable, sys.version))
    report.command("", ["uv", "--version"])


def section_gpu(report):
    if platform.system() == "Darwin":
        return  # MPS facts come from the venv torch probe
    report.heading("GPU and driver")
    report.command("driver + max supported CUDA", ["nvidia-smi"])
    report.command("compute capability", [
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv"])


def section_cuda_toolkit(report):
    report.heading("CUDA toolkit(s) on disk")
    report.command("", ["nvcc", "--version"])
    if platform.system() == "Windows":
        cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        found = sorted(p.name for p in cuda_root.glob("v*")) if cuda_root.exists() else []
        report.note("installed under `%s`: %s" % (cuda_root, found or "none"))
    else:
        found = sorted(str(p) for p in Path("/usr/local").glob("cuda*"))
        report.note("under `/usr/local`: %s" % (found or "none"))


def section_compilers(report):
    report.heading("Compilers")
    system = platform.system()
    if system == "Windows":
        vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if vswhere.exists():
            ok, output = run([str(vswhere), "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                              "-format", "json"])
            installs = []
            if ok:
                try:
                    installs = json.loads(output)
                except ValueError:
                    pass
            for install in installs:
                report.note("- VS install: **%s** (%s) at `%s`" % (
                    install.get("displayName"), install.get("installationVersion"),
                    install.get("installationPath")))
                toolsets = Path(install.get("installationPath", "")) / "VC" / "Tools" / "MSVC"
                if toolsets.exists():
                    report.note("  MSVC toolsets: %s" % sorted(p.name for p in toolsets.iterdir()))
            if not installs:
                report.note("vswhere found no VS install with C++ build tools:\n\n```text\n%s\n```" % output)
        else:
            report.note("vswhere.exe not found — no Visual Studio installer metadata")
        report.command("cl on PATH (usually only in a dev prompt)", "cl 2>&1", shell=True)
    elif system == "Darwin":
        report.command("", ["clang", "--version"])
        report.command("active developer dir", ["xcode-select", "-p"])
    else:
        report.command("", ["gcc", "--version"])
        report.command("", ["g++", "--version"])


def section_modules(report):
    """Environment modules (Lmod etc.) — the HPC CUDA ceiling lives here."""
    if platform.system() == "Windows":
        return
    ok, _ = run(["bash", "-lc", "type module"], timeout=30)
    if not ok:
        return
    report.heading("Environment modules (HPC)")
    report.command("", ["bash", "-lc", "module --version 2>&1"], timeout=60)
    for name in ("cuda", "cudnn", "gcc", "nccl"):
        report.command("", ["bash", "-lc", "module avail %s 2>&1" % name], timeout=120)


def section_slurm(report):
    ok, _ = run(["sinfo", "--version"], timeout=30)
    if not ok:
        return
    report.heading("SLURM")
    report.command("", ["sinfo", "--version"])
    report.command("partitions and GPU GRES", ["sinfo", "-o", "%P %G %D %N"], timeout=60)


def section_venv(report):
    report.heading("Project .venv")
    if platform.system() == "Windows":
        venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        report.note("no `.venv` in the repo — nothing installed here yet; "
                    "`uv.lock` is the reference for what would be installed.")
        return
    # Importing torch can be slow the first time; be generous.
    report.command("venv introspection", [str(venv_python), "-c", VENV_PROBE],
                   timeout=300,
                   shown=".venv python -c <torch + installed-distributions probe>")


def main():
    report = Report()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _ok, commit = run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    report.lines += ["# Platform report: %s (%s)" % (os_key(), platform.node()),
                     "", "Generated %s at commit `%s` by `tools/platform_report.py`." % (
                         now, commit.splitlines()[0] if _ok else "unknown")]

    section_system(report)
    section_python(report)
    section_gpu(report)
    section_cuda_toolkit(report)
    section_compilers(report)
    section_modules(report)
    section_slurm(report)
    section_venv(report)

    hostname = "".join(c if c.isalnum() or c == "-" else "-" for c in platform.node().lower())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / ("%s-%s.md" % (os_key(), hostname or "unknown"))
    out_path.write_text(report.text(), encoding="utf-8")
    print("wrote %s" % out_path)
    print("now: git add, commit and push it so the reports can be compared.")


if __name__ == "__main__":
    main()
