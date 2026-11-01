"""Lazy loader for the Titan CUDA tile rasterizer extension."""

import os
import shutil
import subprocess

import torch
from torch.utils.cpp_extension import load as _torch_load


_CACHE = None


def _find_vcvars64() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.isfile(vswhere):
        try:
            output = subprocess.check_output(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
                errors="replace",
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            output = ""
        if output:
            path = os.path.join(
                output, "VC", "Auxiliary", "Build", "vcvars64.bat"
            )
            if os.path.isfile(path):
                return path
    return None


def _ensure_msvc_env() -> None:
    if shutil.which("cl"):
        return
    vcvars = _find_vcvars64()
    if not vcvars:
        return
    command = f'call "{vcvars}" >nul 2>&1 && set'
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"vcvars64 初始化失败: {result.stderr[:200]}"
        )
    output = result.stdout
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def _ensure_cuda_env() -> None:
    if os.environ.get("CUDA_PATH"):
        return
    version = getattr(torch.version, "cuda", None) or "11.8"
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if not os.path.isdir(base):
        return
    entries = sorted(os.listdir(base), reverse=True)
    exact = f"v{version}"
    chosen = exact if exact in entries else next(
        (entry for entry in entries if entry.lower().startswith("v")),
        None,
    )
    if not chosen:
        return
    cuda_home = os.path.join(base, chosen)
    os.environ["CUDA_PATH"] = cuda_home
    os.environ["CUDA_HOME"] = cuda_home
    bin_dir = os.path.join(cuda_home, "bin")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def tile_cuda():
    """Return the compiled CUDA module, or None when it cannot be built."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not torch.cuda.is_available():
        _CACHE = None
        return None
    try:
        _ensure_msvc_env()
        _ensure_cuda_env()
        capability = torch.cuda.get_device_capability(0)
        os.environ.setdefault(
            "TORCH_CUDA_ARCH_LIST",
            f"{capability[0]}.{capability[1]}",
        )
        source_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(source_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        extra_cuda = ["-O3"]
        extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
        if os.name == "nt":
            # CUDA 11.8 + newer MSVC STL needs this opt-out to build torch headers.
            extra_cuda.append("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
            extra_cuda.append("-allow-unsupported-compiler")
        _CACHE = _torch_load(
            name="titan_tile_cuda",
            sources=[os.path.join(source_dir, "tile_cuda.cu")],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda,
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as exc:
        print(f"[tile_cuda] 编译失败: {exc}")
        _CACHE = None
    return _CACHE
