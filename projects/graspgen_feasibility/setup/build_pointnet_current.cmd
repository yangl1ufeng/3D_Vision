@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "CUDA_HOME=%~dp0..\..\..\envs\graspgen-cuda\Library"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TEMP=%~dp0..\..\..\.downloads\tmp"
set "TMP=%TEMP%"
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=2
set DISTUTILS_USE_SDK=1
cd /d "%~dp0..\third_party\GraspGen\pointnet2_ops"
"%~dp0..\.venv-vision\Scripts\python.exe" -m pip install --no-index --no-build-isolation --no-deps .
