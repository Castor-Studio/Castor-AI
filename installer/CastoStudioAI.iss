; ==============================================================================
; CastoStudio AI Worker — Installateur Windows Professionnel (Inno Setup)
;
; Caractéristiques :
; - Détection matérielle native en PowerShell (zéro faux-positif antivirus).
; - Qualification matérielle CPU/RAM/GPU (Vert / Orange / Rouge).
; - Dépendances IA isolées avec Python 3.12 garanti par uv.
; - Ouverture automatique des règles de Pare-feu Windows (gRPC 50051 & RTMP 1935).
; - Nettoyage complet lors de la désinstallation (règles pare-feu, venv, cache).
; - Contrat avec le front-end via %ProgramData%\CastoStudio\ai_status.json.
; ==============================================================================

#define MyAppName "CastoStudio AI Worker"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "CastoStudio"

[Setup]
; GUID stable pour les mises à jour
AppId={{79489B54-F616-4A94-90CD-1C098291925F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\CastoStudio\AI Worker
DefaultGroupName=CastoStudio AI Worker
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=CastoStudioAI-Setup
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Fichiers sources de l'application (core, server, packages, scripts)
Source: "..\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion; \
    Excludes: ".git\*,.venv\*,__pycache__\*,*.pyc,installer\*,.pytest_cache\*"

; Exécutable uv embarqué (binaire officiel x86_64 pour Windows)
Source: "dist\uv.exe"; DestDir: "{app}"; Flags: ignoreversion

; Scripts PowerShell d'installation et de statut (pas d'exécutable PyInstaller non signé)
Source: "check_resources.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "write_status.ps1"; DestDir: "{tmp}"; Flags: dontcopy
Source: "install_steps.bat"; DestDir: "{tmp}"; Flags: dontcopy

[Run]
; Configuration automatique du Pare-feu Windows Defender pour les communications locales
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""CastoStudio AI Worker (gRPC 50051)"" dir=in action=allow protocol=TCP localport=50051 profile=any"; Flags: runhidden; StatusMsg: "Configuration du pare-feu Windows pour le serveur IA (port 50051)..."
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""CastoStudio MediaMTX Streaming (RTMP 1935)"" dir=in action=allow protocol=TCP localport=1935 profile=any"; Flags: runhidden; StatusMsg: "Configuration du pare-feu Windows pour le streaming local (port 1935)..."

[UninstallRun]
; Nettoyage des règles du pare-feu à la désinstallation
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""CastoStudio AI Worker (gRPC 50051)"""; Flags: runhidden
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""CastoStudio MediaMTX Streaming (RTMP 1935)"""; Flags: runhidden

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: files; Name: "{app}\install.log"

[Code]
var
  ResourcePage: TWizardPage;
  ScoreLabel: TNewStaticText;
  DetailLabel: TNewStaticText;
  MessageLabel: TNewStaticText;
  ConfirmLabel: TNewStaticText;
  ConfirmEdit: TNewEdit;
  ResourceIniFile: string;
  ResourceJsonFile: string;
  ScorePercent: Integer;
  ResourceTier: string;
  UserConfirmedOverride: Boolean;

const
  CONFIRM_PHRASE = 'J ACCEPTE';
  RED_COLOR    = $003C14DC; // Crimson
  ORANGE_COLOR = $00008CFF; // Dark Orange
  GREEN_COLOR  = $00228B22; // Forest Green

procedure RunResourceCheck();
var
  ScriptPath, IniPath, JsonPath: string;
  ResultCode: Integer;
begin
  ScriptPath := ExpandConstant('{tmp}\check_resources.ps1');
  ExtractTemporaryFile('check_resources.ps1');
  IniPath := ExpandConstant('{tmp}\resource_report.ini');
  JsonPath := ExpandConstant('{tmp}\resource_report.json');
  ResourceIniFile := IniPath;
  ResourceJsonFile := JsonPath;

  // Exécution native PowerShell — aucun binaire compilé suspect
  Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + ScriptPath + '" -Ini "' + IniPath + '" -Out "' + JsonPath + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure CreateResourcePage();
begin
  ResourcePage := CreateCustomPage(wpSelectDir, 'Verification de la compatibilite materielle',
    'CastoStudio AI analyse les composants de votre machine pour garantir une analyse fluide en temps reel.');

  ScoreLabel := TNewStaticText.Create(ResourcePage);
  ScoreLabel.Parent := ResourcePage.Surface;
  ScoreLabel.Top := 0;
  ScoreLabel.Left := 0;
  ScoreLabel.AutoSize := True;
  ScoreLabel.Font.Size := 13;
  ScoreLabel.Font.Style := [fsBold];

  DetailLabel := TNewStaticText.Create(ResourcePage);
  DetailLabel.Parent := ResourcePage.Surface;
  DetailLabel.Top := ScoreLabel.Top + 30;
  DetailLabel.Left := 0;
  DetailLabel.AutoSize := True;
  DetailLabel.WordWrap := True;
  DetailLabel.Width := ResourcePage.SurfaceWidth;

  MessageLabel := TNewStaticText.Create(ResourcePage);
  MessageLabel.Parent := ResourcePage.Surface;
  MessageLabel.Top := DetailLabel.Top + 75;
  MessageLabel.Left := 0;
  MessageLabel.AutoSize := False;
  MessageLabel.WordWrap := True;
  MessageLabel.Width := ResourcePage.SurfaceWidth;
  MessageLabel.Height := 55;
  MessageLabel.Font.Style := [fsBold];

  ConfirmLabel := TNewStaticText.Create(ResourcePage);
  ConfirmLabel.Parent := ResourcePage.Surface;
  ConfirmLabel.Top := MessageLabel.Top + 65;
  ConfirmLabel.Left := 0;
  ConfirmLabel.AutoSize := True;
  ConfirmLabel.WordWrap := True;
  ConfirmLabel.Width := ResourcePage.SurfaceWidth;
  ConfirmLabel.Caption := 'Pour forcer l''installation sur ce materiel, tapez exactement : ' + CONFIRM_PHRASE;
  ConfirmLabel.Visible := False;

  ConfirmEdit := TNewEdit.Create(ResourcePage);
  ConfirmEdit.Parent := ResourcePage.Surface;
  ConfirmEdit.Top := ConfirmLabel.Top + 22;
  ConfirmEdit.Left := 0;
  ConfirmEdit.Width := 250;
  ConfirmEdit.Visible := False;
end;

procedure PopulateResourcePage();
var
  CpuCores, RamGb, GpuVram, GpuName: string;
begin
  ScorePercent := StrToIntDef(GetIniString('resources', 'score_percent', '0', ResourceIniFile), 0);
  ResourceTier := GetIniString('resources', 'tier', 'low', ResourceIniFile);
  CpuCores := GetIniString('resources', 'cpu_cores', '?', ResourceIniFile);
  RamGb := GetIniString('resources', 'ram_gb', '?', ResourceIniFile);
  GpuVram := GetIniString('resources', 'gpu_vram_gb', '0', ResourceIniFile);
  GpuName := GetIniString('resources', 'gpu_name', '', ResourceIniFile);
  if GpuName = '' then
    GpuName := 'Aucun GPU dedie detecte';

  ScoreLabel.Caption := 'Score de compatibilite IA : ' + IntToStr(ScorePercent) + ' %';

  DetailLabel.Caption :=
    '• Processeur (CPU) : ' + CpuCores + ' coeurs logiques' + #13#10 +
    '• Memoire vive (RAM) : ' + RamGb + ' Go' + #13#10 +
    '• Carte Graphique (GPU) : ' + GpuName + ' (' + GpuVram + ' Go VRAM)' + #13#10 + #13#10 +
    'Configuration recommandee : 8 coeurs, 16 Go RAM, GPU dedie 6 Go VRAM.';

  if ResourceTier = 'excellent' then
  begin
    ScoreLabel.Font.Color := GREEN_COLOR;
    MessageLabel.Font.Color := GREEN_COLOR;
    MessageLabel.Caption := 'Excellent. Votre machine est parfaitement dimensionnee pour faire tourner l''IA en temps reel.';
    ConfirmLabel.Visible := False;
    ConfirmEdit.Visible := False;
  end
  else if ResourceTier = 'ok' then
  begin
    ScoreLabel.Font.Color := ORANGE_COLOR;
    MessageLabel.Font.Color := ORANGE_COLOR;
    MessageLabel.Caption := 'Ressources correctes. L''IA tournera en local, avec un ralentissement possible lors des charges elevees.';
    ConfirmLabel.Visible := False;
    ConfirmEdit.Visible := False;
  end
  else
  begin
    ScoreLabel.Font.Color := RED_COLOR;
    MessageLabel.Font.Color := RED_COLOR;
    MessageLabel.Caption := 'Ressources faibles pour du traitement video temps reel. L''IA risque d''etre lente ou de surcharger votre machine.';
    ConfirmLabel.Visible := True;
    ConfirmEdit.Visible := True;
    ConfirmEdit.Text := '';
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = ResourcePage.ID) and (ResourceTier = 'low') then
  begin
    UserConfirmedOverride := (Uppercase(Trim(ConfirmEdit.Text)) = CONFIRM_PHRASE);
    if not UserConfirmedOverride then
    begin
      MsgBox('Ressources inferieures aux recommandations.' + #13#10 + #13#10 +
        'Pour continuer l''installation malgre ce risque, tapez exactement "' + CONFIRM_PHRASE + '" dans le champ prevu, ou annulez.',
        mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure InitializeWizard();
begin
  CreateResourcePage();
  UserConfirmedOverride := False;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = ResourcePage.ID then
  begin
    RunResourceCheck();
    PopulateResourcePage();
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  InstallBat, InstallLog, ScriptPath, ConfirmedArg: string;
  ResultCode: Integer;
  InstallOk: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    // 1. Installation des dépendances IA via le script batch (visible pour l'utilisateur)
    InstallLog := ExpandConstant('{app}\install.log');
    InstallBat := ExpandConstant('{tmp}\install_steps.bat');
    ExtractTemporaryFile('install_steps.bat');

    WizardForm.StatusLabel.Caption := 'Installation des composants IA et de PyTorch (veuillez patienter)...';
    // Lancement avec fenêtre visible pour que l'utilisateur voit la progression en direct
    InstallOk := Exec(InstallBat, '"' + ExpandConstant('{app}') + '" "' + InstallLog + '"',
      '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);

    if not InstallOk then
    begin
      MsgBox('L''installation des dependances IA a rencontre une erreur (code ' + IntToStr(ResultCode) + ').' + #13#10 + #13#10 +
        'Consultez le fichier de journal pour plus de details :' + #13#10 + InstallLog + #13#10 + #13#10 +
        'Vous pourrez finaliser l''installation manuellement en executant dans ce dossier :' + #13#10 +
        '".\uv.exe" sync --all-packages --python 3.12',
        mbError, MB_OK);
    end;

    // 2. Ecriture du fichier de statut %ProgramData%\CastoStudio\ai_status.json
    ScriptPath := ExpandConstant('{tmp}\write_status.ps1');
    ExtractTemporaryFile('write_status.ps1');
    if UserConfirmedOverride then
      ConfirmedArg := 'true'
    else
      ConfirmedArg := 'false';

    Exec('powershell.exe',
      '-NoProfile -ExecutionPolicy Bypass -File "' + ScriptPath + '" -Report "' + ResourceJsonFile + '" -Confirmed ' + ConfirmedArg + ' -InstallDir "' + ExpandConstant('{app}') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
