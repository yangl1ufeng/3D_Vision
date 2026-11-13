@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=D:\3D_Vision\envs\graspgen-cuda\Library
set CUDA_PATH=%CUDA_HOME%
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%CUDA_HOME%\bin;%PATH%
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
cd /d D:\3D_Vision\projects\graspgen_feasibility\third_party\GraspGen\pointnet2_ops
D:\3D_Vision\projects\graspgen_feasibility\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps .
