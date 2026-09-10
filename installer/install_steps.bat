@echo off
REM ==============================================================================
REM CastoStudio AI Worker — Script d'installation des dépendances IA
REM Exécute uv sync avec Python 3.12 forcé, génération gRPC et modèles YOLOv8.
REM ==============================================================================

setlocal EnableDelayedExpansion
title CastoStudio AI — Installation des composants IA...

set "APPDIR=%~1"
set "LOGFILE=%~2"
set "UV_EXE=%APPDIR%\uv.exe"

echo === CastoStudio AI install steps === > "%LOGFILE%"
echo Date: %DATE% %TIME% >> "%LOGFILE%"
echo AppDir: %APPDIR% >> "%LOGFILE%"

echo ==============================================================================
echo  CastoStudio AI — Installation en cours...
echo  (Ne fermez pas cette fenetre, cela peut prendre 2 a 5 minutes selon la connexion)
echo ==============================================================================
echo.

cd /d "%APPDIR%"
if errorlevel 1 (
    echo FAILED: Impossible d'acceder au dossier %APPDIR% >> "%LOGFILE%"
    exit /b 1
)

if not exist "%UV_EXE%" (
    echo FAILED: uv.exe introuvable dans %UV_EXE% >> "%LOGFILE%"
    exit /b 1
)

REM ------------------------------------------------------------------------------
REM 1. S'assurer que Python 3.12 autonome est installe par uv
REM ------------------------------------------------------------------------------
echo [1/4] Verification et configuration du runtime Python 3.12...
echo [1/4] Verification et configuration du runtime Python 3.12... >> "%LOGFILE%"
"%UV_EXE%" python install 3.12 >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo AVERTISSEMENT: uv python install 3.12 a retourne un code d'erreur, poursuite avec le Python disponible... >> "%LOGFILE%"
)

REM ------------------------------------------------------------------------------
REM 2. Synchronisation des paquets du workspace (PyTorch, Ultralytics, Core, Podcast)
REM ------------------------------------------------------------------------------
echo [2/4] Installation des dependances IA (PyTorch, Torchvision, Ultralytics)...
echo [2/4] Installation des dependances IA... >> "%LOGFILE%"

set "CACHE_ARG="
if exist "%APPDIR%\cache" (
    set "CACHE_ARG=--cache-dir "%APPDIR%\cache""
    echo Utilisation du cache hors-ligne local >> "%LOGFILE%"
)

"%UV_EXE%" sync --all-packages --python 3.12 %CACHE_ARG% >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo FAILED: uv sync --all-packages --python 3.12 >> "%LOGFILE%"
    echo.
    echo [ERREUR] Echec du telechargement des dependances IA.
    echo Consultez le journal detaille : %LOGFILE%
    exit /b 1
)

REM ------------------------------------------------------------------------------
REM 3. Generation des fichiers gRPC (ia_analysis.proto -> Python)
REM ------------------------------------------------------------------------------
echo [3/4] Generation du code gRPC...
echo [3/4] Generation du code gRPC... >> "%LOGFILE%"
"%UV_EXE%" run python scripts\generate_proto.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo FAILED: generate_proto.py >> "%LOGFILE%"
    echo [ERREUR] Echec de la generation gRPC.
    exit /b 1
)

REM ------------------------------------------------------------------------------
REM 4. Pre-telechargement du modele de vision YOLOv8
REM ------------------------------------------------------------------------------
echo [4/4] Verification et telechargement du modele de vision YOLOv8...
echo [4/4] Verification et telechargement du modele de vision YOLOv8... >> "%LOGFILE%"
"%UV_EXE%" run python scripts\download_models.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo FAILED: download_models.py >> "%LOGFILE%"
    echo [ERREUR] Echec du pre-telechargement du modele YOLOv8.
    exit /b 1
)

echo.
echo ==============================================================================
echo  Installation des composants IA terminee avec succes !
echo ==============================================================================
echo SUCCESS >> "%LOGFILE%"
exit /b 0
