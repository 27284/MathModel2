@echo off
chcp 65001 >nul
cd /d "%~dp0"
"D:\Anaconda\annconda\envs\EEGModel\python.exe" main.py %*
pause
