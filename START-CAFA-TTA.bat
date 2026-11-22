@echo off
setlocal
cd /d "%~dp0"
title CAFA-TTA ALL-IN-ONE - WINDOWS

:MENU
cls
echo ========================================
echo       CAFA-TTA ALL-IN-ONE
echo             WINDOWS 11
echo ========================================
echo.
echo 1. Mo thu muc bai goc
echo 2. Mo ma nguon Docker
echo 3. Mo checkpoints da train
echo 4. Xem ket qua Kather 2016
echo 5. Xem ket qua PathMNIST
echo 6. Xem huong dan du an
echo 7. Thong tin huan luyen Docker
echo 0. Thoat
echo.
set /p CHOICE=Chon chuc nang [0-7]:

if "%CHOICE%"=="1" start "" explorer "%~dp0."
if "%CHOICE%"=="2" start "" explorer "%~dp0docker-project"
if "%CHOICE%"=="3" start "" explorer "%~dp0docker-project\docker_runs"
if "%CHOICE%"=="4" notepad "%~dp0docker-project\docker_runs\kather2016_cpu_01\cafa_comparison_shifted_seed42\comparison_metrics.json"
if "%CHOICE%"=="5" notepad "%~dp0docker-project\docker_runs\baseline_full_cpu_01\test_metrics.json"
if "%CHOICE%"=="6" notepad "%~dp0.\README.md"
if "%CHOICE%"=="7" goto DOCKER
if "%CHOICE%"=="0" exit /b 0

goto MENU

:DOCKER
cls
echo ========================================
echo         DOCKER TRAINING
echo ========================================
echo.
echo Du lieu, checkpoint va ket qua da co tren Windows.
echo Docker training hien dang chay trong Ubuntu VMware.
echo.
echo De train, mo Ubuntu VMware va chay:
echo ~/CAFA-TTA-ALL-IN-ONE/START-CAFA-TTA.sh
echo.
echo Chua cau hinh dieu khien Docker tu Windows.
echo.
pause
goto MENU
