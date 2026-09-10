# Guide de Build — Installateur Windows (Mode Production)

Ce guide décrit comment compiler `CastoStudioAI-Setup.exe` sur une machine Windows 10/11.

---

## Prérequis

1. **Inno Setup 6** :
   ```powershell
   winget install JRSoftware.InnoSetup
   ```
2. **PowerShell 5.1 ou 7+** (intégré à Windows par défaut).
3. **Git** pour récupérer les sources du projet.

*(Note : Python n'est plus requis sur la machine de build car nous utilisons PowerShell pour les scripts de détection matérielle).*

---

## 1. Cloner la branche de production

```powershell
git clone git@github.com:Castor-Studio/Castor-AI.git
cd Castor-AI
git checkout feature/production-installer
```

---

## 2. Préparer les dépendances de build

Dans une invite PowerShell :

```powershell
powershell -ExecutionPolicy Bypass -File .\installer\build_helpers.ps1
```

Ce script télécharge automatiquement l'exécutable officiel `uv.exe` (Windows x64) dans `installer\dist\uv.exe`.

---

## 3. Compiler l'installateur Inno Setup

```powershell
iscc installer\CastoStudioAI.iss
```

*(Si la commande `iscc` n'est pas reconnue dans votre PATH, ouvrez le fichier `installer\CastoStudioAI.iss` directement avec l'application **Inno Setup Compiler** et cliquez sur **Build > Compile**).*

L'installateur est généré dans :
```text
installer\output\CastoStudioAI-Setup.exe
```

---

## 4. Test et validation

Pour tester l'installateur :
1. Lancez `CastoStudioAI-Setup.exe` en mode administrateur.
2. Vérifiez que la page d'évaluation des composants affiche correctement votre CPU, RAM et GPU.
3. Observez l'installation : la console affiche les étapes `[1/4]`, `[2/4]`, etc.
4. À la fin de l'installation, vérifiez la présence du fichier :
   ```powershell
   type "$env:ProgramData\CastoStudio\ai_status.json"
   ```
5. Testez le lancement du serveur gRPC :
   ```powershell
   cd "C:\Program Files\CastoStudio\AI Worker"
   .\uv.exe run castostudio-ai-server --host 127.0.0.1 --port 50051
   ```
