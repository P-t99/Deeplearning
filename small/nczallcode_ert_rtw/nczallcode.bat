
set skipSetupArg=%2
if "%skipSetupArg%" NEQ "skip_setup_msvc" (
call "setup_msvc.bat"
)


call  "\\PC\E$\Program Files\MATLAB\R2024a\bin\win64\checkMATLABRootForDriveMap.exe" "\\PC\E$\Program Files\MATLAB\R2024a"  > mlEnv.txt
for /f %%a in (mlEnv.txt) do set "%%a"\n
cd .

if "%1"=="" (nmake MATLAB_ROOT=%MATLAB_ROOT% ALT_MATLAB_ROOT=%ALT_MATLAB_ROOT% MATLAB_BIN=%MATLAB_BIN% ALT_MATLAB_BIN=%ALT_MATLAB_BIN%  -f nczallcode.mk all) else (nmake MATLAB_ROOT=%MATLAB_ROOT% ALT_MATLAB_ROOT=%ALT_MATLAB_ROOT% MATLAB_BIN=%MATLAB_BIN% ALT_MATLAB_BIN=%ALT_MATLAB_BIN%  -f nczallcode.mk %1)
@if errorlevel 1 goto error_exit

exit /B 0

:error_exit
echo The make command returned an error of %errorlevel%
exit /B 1