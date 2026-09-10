# Installateur Windows Professionnel — CastoStudio AI Worker

Installateur 2-clics moderne (Inno Setup) qui :
1. Analyse les capacités matérielles locales (CPU, RAM, GPU) via des scripts PowerShell natifs (zéro alerte antivirus).
2. Configure automatiquement les règles de **Pare-feu Windows Defender** pour la communication gRPC (port `50051`) et le streaming RTMP (port `1935`).
3. Isole l'environnement d'exécution avec **Python 3.12** garanti via l'outil autonome `uv`.
4. Installe toutes les dépendances IA (`uv sync --all-packages --python 3.12`, génération gRPC, pré-téléchargement YOLOv8) avec affichage de la progression en direct.
5. Écrit le contrat `%ProgramData%\CastoStudio\ai_status.json` pour le front-end CastoStudio.

---

## 🛠️ Build de l'installateur

### Prérequis sur la machine de build
- **Windows 10 ou 11** (x64)
- **PowerShell 5.1+** (intégré à Windows)
- **[Inno Setup 6](https://jrsoftware.org/isdl.php)**

> 💡 **Note importante :** Il n'est plus nécessaire d'installer Python ou PyInstaller sur la machine de build ! Les scripts d'analyse sont exécutés nativement par PowerShell.

### Commandes de compilation

```powershell
# 1. Prépare uv.exe (téléchargement unique)
powershell -ExecutionPolicy Bypass -File .\installer\build_helpers.ps1

# 2. Compile l'installateur avec Inno Setup
iscc installer\CastoStudioAI.iss
```

L'exécutable final sera généré dans :
```text
installer\output\CastoStudioAI-Setup.exe
```

---

## 📦 Mode Hors-Ligne / Salons (Optionnel)

Si vous devez installer CastoStudio AI sur des PC lors d'une démonstration sans connexion Internet ou avec un Wi-Fi public instable (ex: salon, Epitech Experience) :

```powershell
# Pré-télécharge toutes les roues PyTorch/CUDA dans installer\cache\
powershell -ExecutionPolicy Bypass -File .\installer\build_helpers.ps1 -Offline
iscc installer\CastoStudioAI.iss
```
L'installateur installera alors tous les packages en 20 secondes chrono sans aucun appel réseau.

---

## 🔍 Ce que fait l'installateur en détail

1. **Page de bienvenue & Choix du dossier d'installation** (par défaut : `C:\Program Files\CastoStudio\AI Worker`).
2. **Page de compatibilité matérielle** :
   - Exécute `check_resources.ps1` nativement.
   - Calcule un score 0-100% :
     - CPU (30%) : 4 cœurs min / 8 cœurs recommandés.
     - RAM (30%) : 8 Go min / 16 Go recommandés.
     - GPU VRAM (40%) : Détecte les GPU NVIDIA / AMD (6 Go VRAM recommandés).
   - Affiche le résultat avec un code couleur :
     - **Vert (≥80%)** : "Excellent, tournera de façon fluide."
     - **Orange (50-79%)** : "Correct mais pas optimal, ralentissement possible."
     - **Rouge (<50%)** : "Ressources insuffisantes." + **zone de confirmation obligatoire** (l'utilisateur doit taper `J ACCEPTE` pour continuer).
3. **Configuration du Pare-feu Windows Defender** :
   - Règle entrante TCP pour le port `50051` (`CastoStudio AI Worker`).
   - Règle entrante TCP pour le port `1935` (`CastoStudio MediaMTX Streaming`).
   - Nettoyées automatiquement lors de la désinstallation.
4. **Installation des dépendances IA** :
   - Exécute `install_steps.bat` avec une fenêtre de progression visible (`[1/4]`, `[2/4]`, etc.).
   - Force **Python 3.12** pour garantir la compatibilité avec PyTorch et Ultralytics (évite tout conflit avec Python 3.14).
   - Journalise toutes les étapes dans `{app}\install.log`.
5. **Écriture du contrat JSON** :
   - Crée `%ProgramData%\CastoStudio\ai_status.json`.

---

## 📄 Contrat `ai_status.json`

```json
{
  "version": 1,
  "checked_at": "2026-09-10T12:00:00Z",
  "install_dir": "C:\\Program Files\\CastoStudio\\AI Worker",
  "cpu_cores": 8,
  "ram_gb": 16.0,
  "gpu_name": "NVIDIA GeForce RTX 4070",
  "gpu_vram_gb": 8.0,
  "score_percent": 95,
  "tier": "excellent",
  "recommended": true,
  "user_confirmed_override": false,
  "ai_ready": true
}
```

Le front-end C# lit principalement `ai_ready: true|false` pour activer ou masquer les fonctionnalités IA locales.

---

## 🚀 Lancement manuel du serveur IA par le front-end

Une fois installé, le front-end lance le sous-processus suivant :

```powershell
cd "<install_dir>"
.\uv.exe run castostudio-ai-server --host 127.0.0.1 --port 50051
```
