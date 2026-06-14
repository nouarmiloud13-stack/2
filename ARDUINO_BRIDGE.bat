@echo off
chcp 65001 > nul
title GNL — Arduino Serial Bridge

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║         GNL — ARDUINO SERIAL BRIDGE                     ║
echo  ║         Arduino COM3  →  MQTT broker.hivemq.com         ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM ── Verifier Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERREUR: Python non trouve. Installez Python 3.10+ depuis python.org
    pause
    exit /b 1
)

REM ── Installer pyserial et paho-mqtt si absents ────────────────────────────────
echo  Verification des dependances...
python -c "import serial, paho" >nul 2>&1
if errorlevel 1 (
    echo  Installation de pyserial et paho-mqtt...
    pip install pyserial paho-mqtt --quiet
    if errorlevel 1 (
        pip install pyserial paho-mqtt --user --quiet
    )
    echo  Dependances installees.
) else (
    echo  Dependances OK.
)

REM ── Lancement du bridge ───────────────────────────────────────────────────────
echo.
echo  Connexion Arduino sur COM3  →  broker.hivemq.com:1883
echo  (Ctrl+C pour arreter)
echo.

python arduino_serial_bridge.py --port COM3 --host broker.hivemq.com --public

echo.
echo  Bridge arrete.
pause
