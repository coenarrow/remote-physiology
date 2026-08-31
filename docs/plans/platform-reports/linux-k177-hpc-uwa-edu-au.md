# Platform report: linux (k177.hpc.uwa.edu.au)

Generated 2026-08-31 05:59 UTC at commit `4b0bd06` by `tools/platform_report.py`.

## System

- platform: `Linux-5.14.0-503.38.1.el9_5.x86_64-x86_64-with-glibc2.34`
- machine: `x86_64`
- hostname: `k177.hpc.uwa.edu.au`

**`cat /etc/os-release`** — distro

```text
NAME="Rocky Linux"
VERSION="9.5 (Blue Onyx)"
ID="rocky"
ID_LIKE="rhel centos fedora"
VERSION_ID="9.5"
PLATFORM_ID="platform:el9"
PRETTY_NAME="Rocky Linux 9.5 (Blue Onyx)"
ANSI_COLOR="0;32"
LOGO="fedora-logo-icon"
CPE_NAME="cpe:/o:rocky:rocky:9::baseos"
HOME_URL="https://rockylinux.org/"
VENDOR_NAME="RESF"
VENDOR_URL="https://resf.org/"
BUG_REPORT_URL="https://bugs.rockylinux.org/"
SUPPORT_END="2032-05-31"
ROCKY_SUPPORT_PRODUCT="Rocky-Linux-9"
ROCKY_SUPPORT_PRODUCT_VERSION="9.5"
REDHAT_SUPPORT_PRODUCT="Rocky Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="9.5"
```

**`ldd --version | head -1`** — glibc floor for manylinux wheels

```text
ldd (GNU libc) 2.34
```


## Python and uv

script interpreter: `/mmfs1/data/group/pgh004/carrow/repo/remote-physiology/.uv_cache/environments-v2/platform-report-9cc5a911573776d1/bin/python`

```text
3.13.7 (main, Aug 18 2025, 19:20:03) [Clang 20.1.4 ]
```

**`uv --version`**

```text
uv 0.9.16
```


## GPU and driver

**`nvidia-smi`** — driver + max supported CUDA

```text
Mon Aug 31 13:59:33 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:01:00.0 Off |                    0 |
| N/A   31C    P0             55W /  400W |       0MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

**`nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv`** — compute capability

```text
name, driver_version, memory.total [MiB], compute_cap
NVIDIA A100-SXM4-40GB, 575.57.08, 40960 MiB, 8.0
```


## CUDA toolkit(s) on disk

**`nvcc --version`**

```text
command not found
```

under `/usr/local`: none


## Compilers

**`gcc --version`**

```text
gcc (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

**`g++ --version`**

```text
g++ (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```


## Environment modules (HPC)

**`bash -lc module --version 2>&1`**

```text
Modules based on Lua: Version 8.7.65 2025-08-05 10:24 -06:00
    by Robert McLay mclay@tacc.utexas.edu
```

**`bash -lc module avail cuda 2>&1`**

```text
-------------------------------------------------------------- /uwahpc/rocky9/modulefiles/devel --------------------------------------------------------------
   cuda/12.4.1    cuda/12.6.3 (D)    nvhpc-hpcx-cuda12/25.5

  Where:
   D:  Default Module

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```

**`bash -lc module avail cudnn 2>&1`**

```text
No module(s) or extension(s) found!
If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```

**`bash -lc module avail gcc 2>&1`**

```text
-------------------------------------------------------------- /uwahpc/rocky9/modulefiles/devel --------------------------------------------------------------
   gcc/11.5.0    gcc/12.4.0    gcc/14.3.0 (D)

  Where:
   D:  Default Module

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```

**`bash -lc module avail nccl 2>&1`**

```text
No module(s) or extension(s) found!
If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
```


## SLURM

**`sinfo --version`**

```text
slurm 24.11.5
```

**`sinfo -o %P %G %D %N`** — partitions and GPU GRES

```text
PARTITION GRES NODES NODELIST
ondemand (null) 1 k011
ondemand-gpu gpu:v100-32gb:2 4 k[017-020]
ondemand-gpu gpu:v100:2 18 k[027-044]
amdgpu gpu:mi210:2 2 k[014-015]
work* (null) 12 k[001-010,012-013]
gpu gpu:v100-32gb:2 4 k[017-020]
gpu gpu:v100:2 18 k[027-044]
medical gpu:h100:4 1 k178
pophealth gpu:v100-32gb:2 1 k171
pophealth gpu:a100:4 1 k177
```


## Project .venv

no `.venv` in the repo — nothing installed here yet; `uv.lock` is the reference for what would be installed.
