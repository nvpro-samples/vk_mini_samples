@echo off

if not exist build (
    mkdir build
)

cd build
rem cmake -DCMAKE_BUILD_TYPE=Release .. -A x64
cmake --build . --config Release --parallel 10

echo.
set /P M=Do you want to build DEBUG configuration? [y/n]
if /I "%M%" neq "y" goto END

:BUILD_DEBUG

cmake -DCMAKE_BUILD_TYPE=Debug .. -A x64
cmake --build . --config Debug --parallel 10

:END
pause
