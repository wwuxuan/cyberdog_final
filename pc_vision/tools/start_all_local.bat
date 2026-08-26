@echo off
setlocal

if "%~1"=="" (
  echo Usage: %~nx0 ^<DOG_IP^>
  echo Example: %~nx0 192.168.43.247
  exit /b 2
)

set "DOG_IP=%~1"
set "ROOT=%~dp0.."
set "PYTHON=python"

start "CyberDog Stage 1 Fisheye :9876" cmd /k ""%PYTHON%" "%ROOT%\stage1_vision.py" --dog-ip "%DOG_IP%""
start "CyberDog Stage 3 Fisheye :9877" cmd /k ""%PYTHON%" "%ROOT%\stage3_vision.py" --dog-ip "%DOG_IP%""
start "CyberDog Stage 4 YOLO :9891" cmd /k ""%PYTHON%" "%ROOT%\stage4_yolo.py" --port 9891 --dog-ip "%DOG_IP%" --push-ip "%DOG_IP%""
start "CyberDog Stage 6 Football YOLO :9892" cmd /k ""%PYTHON%" "%ROOT%\stage6_yolo.py" --port 9892 --push-ip "%DOG_IP%""

echo Started four local services for dog %DOG_IP%.
endlocal
