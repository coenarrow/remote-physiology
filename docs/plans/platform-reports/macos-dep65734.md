# Platform report: macos (dep65734)

Generated 2026-08-31 05:56 UTC at commit `02746eb` by `tools/platform_report.py`.

## System

- platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
- machine: `arm64`
- hostname: `dep65734`

**`sw_vers`** — macOS version

```text
ProductName:		macOS
ProductVersion:		26.6.2
BuildVersion:		25G83
```

**`sysctl -n machdep.cpu.brand_string`** — chip

```text
Apple M1 Pro
```


## Python and uv

script interpreter: `/Users/20759193/repos/remote-physiology/.uv_cache/environments-v2/platform-report-95597b08a94a6770/bin/python`

```text
3.13.6 (main, Aug 14 2025, 16:07:26) [Clang 20.1.4 ]
```

**`uv --version`**

```text
uv 0.10.9 (Homebrew 2026-03-06)
```


## CUDA toolkit(s) on disk

**`nvcc --version`**

```text
command not found
```

under `/usr/local`: none


## Compilers

**`clang --version`**

```text
Apple clang version 21.0.0 (clang-2100.1.1.101)
Target: arm64-apple-darwin25.6.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

**`xcode-select -p`** — active developer dir

```text
/Applications/Xcode.app/Contents/Developer
```


## Project .venv

no `.venv` in the repo — nothing installed here yet; `uv.lock` is the reference for what would be installed.
