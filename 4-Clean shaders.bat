@echo off

pushd ..
FOR /D /R %%G IN ("*") DO (  rem Iterate through all subfolders
  IF NOT %%G == ".git" CD %%G
  IF EXIST _autogen (
    rd /q /s _autogen
    echo "Deleted in " %%G
  )
)
popd
pause