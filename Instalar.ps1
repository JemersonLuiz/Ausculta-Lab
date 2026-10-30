param()
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($PSScriptRoot.Length -gt 90) {
    throw 'A pasta do projeto e longa demais para instalar o PyTorch neste Windows. Extraia o ZIP em uma pasta curta, por exemplo C:\AuscultaLab ou uma pasta AuscultaLab diretamente no seu perfil de usuario, e execute INSTALAR.bat novamente. Nenhuma configuracao do Windows foi alterada.'
}
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv (Join-Path $PSScriptRoot '.venv')
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $PSScriptRoot '.venv')
    } else {
        throw 'Instale Python 3.13 de 64 bits e execute novamente.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente Python.' }
}
Write-Host 'Instalando as dependencias verificadas. A primeira instalacao requer internet.'
& $pythonPath -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Falha na instalacao das dependencias. Consulte a mensagem acima.' }
& $pythonPath -m pip install --disable-pip-version-check --no-deps -e $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Falha na instalacao do pacote local.' }
Write-Host 'Instalacao concluida. Execute INICIAR.bat.'

