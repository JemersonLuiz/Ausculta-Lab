param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { & (Join-Path $PSScriptRoot 'Instalar.ps1') }
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Escolha uma porta entre 1024 e 65535.' }
$previewUrl = "http://127.0.0.1:$Port"
Write-Host "Ausculta Lab - $previewUrl"
Write-Host 'Mantenha esta janela aberta. Use Ctrl+C para encerrar o servidor.'
if (-not $NoBrowser) {
    # A tarefa abre apenas o navegador solicitado; o servidor permanece visivel nesta janela.
    $browserJob = Start-Job -ArgumentList $previewUrl -ScriptBlock {
        param($targetUrl)
        for ($attempt = 0; $attempt -lt 40; $attempt++) {
            try {
                $response = Invoke-WebRequest -Uri "$targetUrl/api/status" -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) { Start-Process $targetUrl; break }
            } catch { Start-Sleep -Milliseconds 500 }
        }
    }
}
try { & $pythonPath (Join-Path $PSScriptRoot 'run.py') --port $Port }
finally {
    if ($browserJob) { Stop-Job -Job $browserJob -ErrorAction SilentlyContinue; Remove-Job -Job $browserJob -ErrorAction SilentlyContinue }
}
if ($LASTEXITCODE -ne 0) { throw 'O servidor nao iniciou ou foi encerrado com erro. Verifique a porta e as dependencias.' }

