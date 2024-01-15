@echo off
setlocal enabledelayedexpansion

set "targetFolder=%CD%"  REM Set the target folder to the current directory or provide a specific path

echo Deleting _autogen folders recursively in %targetFolder%...

for /r "%targetFolder%" %%d in (_autogen) do (
    if exist "%%d" (
        echo Deleting: "%%d"
        rmdir /s /q "%%d"
    )
)

echo Deletion completed.

endlocal

pause