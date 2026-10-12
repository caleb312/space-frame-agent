@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 空间结构 Agent 工作台
set "framePythonArgs="
py -3 -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>&1
if not errorlevel 1 (
  set "framePython=py"
  set "framePythonArgs=-3"
  goto launch
)
python -c "import sys; sys.exit(sys.version_info < (3,10))" >nul 2>&1
if not errorlevel 1 (
  set "framePython=python"
  goto launch
)
set "framePython=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%framePython%" goto launch
echo 未找到 Python 3.10 或更新版本。
echo 请先安装 Python：https://www.python.org/downloads/windows/
echo 安装时勾选 Add Python to PATH，然后重新双击本文件。
pause
exit /b 1

:launch
"%framePython%" %framePythonArgs% launch_workbench.py %*
set "frameExit=%errorlevel%"
if "%frameExit%"=="0" exit /b 0
echo.
echo 启动未完成。请查看上方提示，详细记录位于 .local\startup.log。
pause
exit /b %frameExit%
