; Inno Setup script for RefEye — wraps the PyInstaller folder build
; (dist/RefEye/, produced by `python -m deployment.packaging.build`) into a
; single Setup.exe: the client double-clicks it, clicks Next a few times,
; and gets a Start Menu entry + optional desktop shortcut. No Python, no
; source code, no manual file copying.
;
; Installs per-user (no admin/UAC prompt) so config/ and logs/ beside the
; exe — which the app writes to at runtime (architecture.md section 53) —
; stay writable without elevation.
;
; Build:
;   1. python -m deployment.packaging.build     (produces dist\RefEye\)
;   2. ISCC deployment\packaging\RefEye.iss     (produces dist\installer\RefEye-Setup.exe)

#define MyAppName "RefEye"
#define MyAppVersion "0.1.0-trial"
#define MyAppPublisher "RefEye"
#define MyAppExeName "RefEye.exe"

[Setup]
; Fixed AppId — keep this stable across builds so a re-install upgrades in
; place instead of creating a second entry in "Apps & Features".
AppId={{DBE1D0A7-CD30-4A2C-826A-D2EECC6CF50C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist\installer
OutputBaseFilename=RefEye-Setup
SetupIconFile=refeye.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableWelcomePage=no
; The source tree can top a few hundred MB (model weights + demo clips) —
; this keeps ISCC's own memory/temp footprint reasonable while compressing.
LZMAUseSeparateProcess=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
; Everything PyInstaller produced — exe, runtime, config, models, demo
; videos, logs dir — copied as-is, nothing recompiled or reprocessed.
Source: "..\..\dist\RefEye\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent
