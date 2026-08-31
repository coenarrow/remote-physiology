# Platform report: windows (DEP51484)

Generated 2026-08-31 05:54 UTC at commit `d95611c` by `tools/platform_report.py`.

## System

- platform: `Windows-10-10.0.19045-SP0`
- machine: `AMD64`
- hostname: `DEP51484`

- env `CUDA_PATH` = `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`


## Python and uv

script interpreter: `C:\Users\20759193\source\repos\remote-physiology\.uv_cache\environments-v2\platform-report-fbc2d3c079d4c649\Scripts\python.exe`

```text
3.13.7 (main, Aug 18 2025, 19:16:27) [MSC v.1944 64 bit (AMD64)]
```

**`uv --version`**

```text
uv 0.8.12 (36151df0e 2025-08-18)
```


## GPU and driver

**`nvidia-smi`** — driver + max supported CUDA

```text
Mon Aug 31 13:54:05 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 596.36                 Driver Version: 596.36         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 5000              WDDM  |   00000000:01:00.0 Off |                  Off |
| 34%   29C    P8             14W /  230W |     303MiB /  16384MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            8184    C+G   ..._pzs8sxrjxfjjc\app\claude.exe      N/A      |
|    0   N/A  N/A           12896    C+G   ...0_x64__8j3eq9eme6ctt\IGCC.exe      N/A      |
|    0   N/A  N/A           14848    C+G   ...h_cw5n1h2txyewy\SearchApp.exe      N/A      |
|    0   N/A  N/A           17632    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           17676    C+G   ..._pzs8sxrjxfjjc\app\claude.exe      N/A      |
|    0   N/A  N/A           17696    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           28728    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A           42216    C+G   ...ms\Microsoft VS Code\Code.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

**`nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv`** — compute capability

```text
name, driver_version, memory.total [MiB], compute_cap
Quadro RTX 5000, 596.36, 16384 MiB, 7.5
```


## CUDA toolkit(s) on disk

**`nvcc --version`**

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```

installed under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`: ['v11.1', 'v11.6', 'v11.8', 'v12.1', 'v12.4', 'v12.6']


## Compilers

- VS install: **Visual Studio Community 2022** (17.14.36414.22) at `C:\Program Files\Microsoft Visual Studio\2022\Community`

  MSVC toolsets: ['14.44.35207']

- VS install: **Visual Studio Build Tools 2022** (17.14.36414.22) at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`

  MSVC toolsets: ['14.44.35207']

**`cl 2>&1`** — cl on PATH (usually only in a dev prompt)

```text
'cl' is not recognized as an internal or external command,
operable program or batch file.
[exit code 1]
```


## Project .venv

**`.venv python -c <torch + installed-distributions probe>`** — venv introspection

```text
python: 3.13.7 (main, Aug 18 2025, 19:16:27) [MSC v.1944 64 bit (AMD64)]
torch: 2.6.0+cu124
torch.version.cuda: 12.4
torch.cuda.is_available(): True
torch.cuda.get_arch_list(): ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
cudnn: 90100
gpu 0 Quadro RTX 5000 capability (7, 5)
mps available: False
--- installed distributions (107) ---
asttokens==3.0.0
black==25.1.0
causal_conv1d==1.7.0
certifi==2025.8.3
cffi==2.0.0
charset-normalizer==3.4.3
click==8.2.1
colorama==0.4.6
comm==0.2.3
contourpy==1.3.3
cryptography==46.0.3
cycler==0.12.1
debugpy==1.8.16
decorator==5.2.1
donfig==0.8.1.post1
einops==0.8.1
executing==2.2.1
filelock==3.19.1
fonttools==4.59.2
fsspec==2025.7.0
google-crc32c==1.8.0
h5py==3.14.0
huggingface-hub==0.34.4
idna==3.10
imageio==2.37.0
iniconfig==2.1.0
ipykernel==6.30.1
ipympl==0.9.8
ipython==9.5.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.7
isort==6.0.1
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.2
jupyter_client==8.6.3
jupyter_core==5.8.1
jupyterlab_widgets==3.0.15
kiwisolver==1.4.9
lazy_loader==0.4
mamba_ssm==2.3.1
MarkupSafe==3.0.2
matplotlib==3.10.6
matplotlib-inline==0.1.7
mpmath==1.3.0
mypy_extensions==1.1.0
nest-asyncio==1.6.0
networkx==3.5
neurokit2==0.2.12
ninja==1.13.0
numcodecs==0.16.5
numpy==2.3.2
opencv-python==4.11.0.86
packaging==25.0
pandas==2.3.2
parso==0.8.5
pathspec==0.12.1
pillow==11.3.0
platformdirs==4.4.0
pluggy==1.6.0
prompt_toolkit==3.0.52
protobuf==6.32.0
psutil==7.0.0
pure_eval==0.2.3
pycparser==2.22
Pygments==2.19.2
PyMuPDF==1.26.6
pyparsing==3.2.3
pypdf==6.4.0
pytest==8.4.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyWavelets==1.9.0
pywin32==311
PyYAML==6.0.2
pyzmq==27.0.2
regex==2025.9.1
requests==2.32.5
safetensors==0.6.2
scikit-image==0.25.2
scikit-learn==1.7.1
scipy==1.16.1
seaborn==0.13.2
setuptools==80.9.0
six==1.17.0
stack-data==0.6.3
sympy==1.13.1
tensorboardX==2.6.4
thop==0.1.1-2209072238
threadpoolctl==3.6.0
tifffile==2025.8.28
timm==1.0.19
tokenizers==0.22.0
torch==2.6.0+cu124
torchvision==0.21.0+cu124
tornado==6.5.2
tqdm==4.67.1
traitlets==5.14.3
transformers==4.56.1
triton-windows==3.2.0.post21
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
wcwidth==0.2.13
widgetsnbextension==4.0.14
yacs==0.1.8
zarr==3.3.0
```
