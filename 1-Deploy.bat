@echo off
setlocal enabledelayedexpansion 
set nc[0]="../nvpro_core"
set nc[1]="../../nvpro_core"
set nc[2]="./nvpro_core"

set found=0
for %%a in (0,1,2) do ( 
    if exist !nc[%%a]! (
        echo Found nvpro-sample at: !nc[%%a]!
        set found=1
        set nvpro_core_path=!nc[%%a]!
    )
)

if not %found% == 1 (
    echo Cloning nvpro-sample
    git clone https://github.com/nvpro-samples/nvpro_core.git --recurse-submodules --shallow-submodules
) else (
    echo Updating %nvpro_core_path%
    pushd %nvpro_core_path%
    git submodule update --init --recursive
    popd
)

@REM Updating current 
git submodule update --init --recursive

mkdir build
cd build
del CMakeCache.txt
cmake .. -A x64
if %ERRORLEVEL% neq 0 call :ErrorOccured

pause
exit /b 0

:ErrorOccured

echo.
echo %~1
echo.

pause

exit /b 1
