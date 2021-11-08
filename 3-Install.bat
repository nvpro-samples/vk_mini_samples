@echo off

if not exist _Install (
    mkdir _Install
)

cmake --install _Build --prefix _Install

set /P M=Do you want to build DEBUG configuration? [y/n]
if /I "%M%" neq "y" goto END

:BUILD_DEBUG
cmake --install _Build --prefix _Install --config Debug

:END
pause
