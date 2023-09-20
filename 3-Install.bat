@echo off

if not exist _install (
    mkdir _install
)

cmake --install build --prefix _install

set /P M=Do you want to build DEBUG configuration? [y/n]
if /I "%M%" neq "y" goto END

:BUILD_DEBUG
cmake --install build --prefix _install --config Debug

:END
pause
