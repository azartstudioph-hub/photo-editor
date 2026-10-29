@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Перетягніть сюди папку з фото і натисніть Enter:
set /p FOLDER=
python app.py "%FOLDER%"
pause
