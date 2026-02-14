; ReduxLab-Setup.iss
; Script de Inno Setup para ReduxLab (per-user, sin admin)

[Setup]
AppId={{8E7F2A1B-4C3D-4E5F-8901-PCA234567890}}
AppName=ReduxLab
AppVersion=2.1.0
AppVerName=ReduxLab 2.1.0
AppPublisher=Instituto de Investigaciones Económicas UNAM
AppPublisherURL=https://github.com/daardavid/PCA-SS
AppSupportURL=https://github.com/daardavid/PCA-SS/issues
AppUpdatesURL=https://github.com/daardavid/PCA-SS/releases
AppCopyright=Copyright (C) 2024 David Armando Abreu Rosique

; === Instalación per-user (sin admin) ===
DefaultDirName={localappdata}\Programs\ReduxLab
DefaultGroupName=ReduxLab
AllowNoIcons=yes
; Actualizar identificadores de arquitectura (x64 deprecado). x64compatible permite instalación en WOW64 si aplica.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; === Recursos (opcionales y con ruta absoluta al .iss) ===
#ifexist "{#SourcePath}\LICENSE.txt"
LicenseFile={#SourcePath}\LICENSE.txt
#endif
#ifexist "{#SourcePath}\README.md"
InfoBeforeFile={#SourcePath}\README.md
#endif
#ifexist "{#SourcePath}\app_icon.ico"
SetupIconFile={#SourcePath}\app_icon.ico
#endif

UninstallDisplayIcon={app}\ReduxLab.exe
OutputDir=Output
OutputBaseFilename=ReduxLab-Setup-2.1.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
; Eliminado OnlyBelowVersion obsoleto. Quick Launch es legado en Windows 10+, se mantiene opción solo si el usuario lo desea.
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "Asociar archivos .json de proyectos PCA"; GroupDescription: "Asociaciones de archivo:"

[Files]
; Solo la app empaquetada por PyInstaller
Source: "{#SourcePath}\dist\ReduxLab\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Documentación (opcionales)
Source: "{#SourcePath}\README.md";   DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SourcePath}\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme skipifsourcedoesntexist
Source: "{#SourcePath}\THIRD_PARTY_LICENSES.txt"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

; (Se eliminan: "Fortun(e) 500", "Programa Socioeconómicos" y "Proyectos save")

[Registry]
; Asociación de archivos .pcaproject (HKA = HKCU per-user)
Root: HKA; Subkey: "Software\Classes\.pcaproject"; ValueType: string; ValueName: ""; ValueData: "PCAProject"; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\PCAProject"; ValueType: string; ValueName: ""; ValueData: "Proyecto ReduxLab"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\PCAProject\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\ReduxLab.exe,0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\PCAProject\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\ReduxLab.exe"" ""%1"""; Tasks: associate

[Icons]
Name: "{group}\ReduxLab"; Filename: "{app}\ReduxLab.exe"; WorkingDir: "{app}"
#ifexist "{#SourcePath}\README.md"
Name: "{group}\Manual de Usuario"; Filename: "{app}\README.md"
#endif
; (Se elimina acceso directo "Ejemplos" porque ya no se instalan datos)
Name: "{group}\{cm:UninstallProgram,ReduxLab}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ReduxLab"; Filename: "{app}\ReduxLab.exe"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\ReduxLab"; Filename: "{app}\ReduxLab.exe"; WorkingDir: "{app}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\ReduxLab.exe"; Description: "{cm:LaunchProgram,ReduxLab}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\ReduxLab"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;

  if not IsWin64 then begin
    MsgBox('Esta aplicación requiere Windows de 64 bits.', mbError, MB_OK);
    Result := False;
    Exit;
  end;

  if not (GetWindowsVersion >= $0A000000) then begin
    if MsgBox('Esta aplicación está optimizada para Windows 10 o superior. ¿Desea continuar?', 
              mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    CreateDir(ExpandConstant('{userappdata}\ReduxLab'));
    CreateDir(ExpandConstant('{userappdata}\ReduxLab\projects'));
    CreateDir(ExpandConstant('{userappdata}\ReduxLab\logs'));
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
  if MsgBox('¿Está seguro de que desea desinstalar ReduxLab?'#13#10'Se conservarán sus proyectos guardados en Documentos.',
            mbConfirmation, MB_YESNO) = IDNO then
    Result := False;
end;
