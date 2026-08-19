<#
.SYNOPSIS
    Validação E2E da Parte 1 do teste técnico.

.DESCRIPTION
    Este script organiza em um único fluxo as evidências técnicas que
    queremos ter antes de considerar a Parte 1 encerrada:

      1. testes automatizados;
      2. validação do docker-compose.yml;
      3. build e subida do serviço via Docker Compose;
      4. health check;
      5. requisição real ao endpoint JSON;
      6. requisição real ao endpoint PNG;
      7. build multi-arquitetura amd64/arm64 via Buildx.

    O script não substitui a demonstração da Parte 3. Ele reduz o risco
    de chegar à banca com um requisito que não foi executado de ponta a
    ponta no ambiente local.

.PARAMETER Port
    Porta do host. O container continua usando a porta 8000.
    O padrão aqui é 8001 porque no ambiente de desenvolvimento já houve
    conflito local na porta 8000.

.PARAMETER SkipMultiArch
    Pula apenas a repetição do build multiarch. Útil durante iterações
    rápidas. Para a validação final, execute SEM este switch.

.PARAMETER KeepRunning
    Mantém o Docker Compose ativo ao final para inspeção manual/Swagger.
#>

param(
    [int]$Port = 8001,
    [switch]$SkipMultiArch,
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Message
    Write-Host "============================================================"
}

function Assert-LastExitCode {
    param([string]$Context)

    if ($LASTEXITCODE -ne 0) {
        throw "$Context falhou com exit code $LASTEXITCODE."
    }
}

# -----------------------------------------------------------------
# Resolve a raiz do projeto a partir da localização deste script.
# scripts/verify_part1.ps1 -> raiz = ..
# -----------------------------------------------------------------
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$env:APP_PORT = "$Port"

$HealthUrl = "http://127.0.0.1:$Port/health"
$InferUrl = "http://127.0.0.1:$Port/api/v1/infer"
$AnnotatedUrl = "http://127.0.0.1:$Port/api/v1/infer/annotated"

$InputImage = "samples/input.jpg"
$EvidenceJson = "outputs/evidence-infer.json"
$EvidencePng = "outputs/evidence-annotated.png"

try {
    # =============================================================
    # 1. TESTES AUTOMATIZADOS
    # =============================================================
    Write-Step "1/7 - Testes automatizados"

    python -m pytest -v
    Assert-LastExitCode "pytest"

    # =============================================================
    # 2. VALIDAÇÃO SINTÁTICA DO COMPOSE
    # =============================================================
    Write-Step "2/7 - Validação do docker-compose.yml"

    docker compose config --quiet
    Assert-LastExitCode "docker compose config"

    # =============================================================
    # 3. BUILD E SUBIDA DO SERVIÇO
    # =============================================================
    Write-Step "3/7 - Build e execução via Docker Compose"

    docker compose up --build -d
    Assert-LastExitCode "docker compose up"

    # =============================================================
    # 4. HEALTH CHECK COM RETRY
    # =============================================================
    Write-Step "4/7 - Health check"

    $Healthy = $false

    for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
        try {
            $Health = Invoke-RestMethod `
                -Uri $HealthUrl `
                -Method Get `
                -TimeoutSec 3

            if ($Health.status -eq "ok") {
                $Healthy = $true
                break
            }
        }
        catch {
            # O modelo pode levar alguns segundos para ficar pronto.
            Start-Sleep -Seconds 2
        }
    }

    if (-not $Healthy) {
        docker compose logs --tail 100
        throw "A API não ficou saudável no tempo esperado."
    }

    Write-Host "Health OK: $($Health | ConvertTo-Json -Compress)"

    # =============================================================
    # 5. ENDPOINT JSON REAL
    # =============================================================
    Write-Step "5/7 - Endpoint JSON real"

    $JsonText = & curl.exe `
        -sS `
        -X POST `
        $InferUrl `
        -F "file=@$InputImage"

    Assert-LastExitCode "curl endpoint JSON"

    $JsonText | Set-Content `
        -Path $EvidenceJson `
        -Encoding utf8

    $Json = $JsonText | ConvertFrom-Json

    if ($Json.persons_detected -lt 1) {
        throw "O endpoint JSON não detectou pessoas na imagem de evidência."
    }

    if (-not $Json.alert) {
        throw "O endpoint JSON não retornou o objeto de alerta."
    }

    # A imagem de demonstração validada no projeto contém uma pessoa
    # em zona vermelha. Por isso esperamos o alerta global CRITICAL.
    if ($Json.alert.level -ne "CRITICAL") {
        throw (
            "Alerta global inesperado. " +
            "Esperado=CRITICAL Obtido=$($Json.alert.level)"
        )
    }

    Write-Host "JSON OK -> $EvidenceJson"
    Write-Host "Pessoas detectadas: $($Json.persons_detected)"
    Write-Host "Alerta global: $($Json.alert.level) / $($Json.alert.action)"

    # =============================================================
    # 6. ENDPOINT PNG REAL
    # =============================================================
    Write-Step "6/7 - Endpoint PNG real"

    & curl.exe `
        -sS `
        -X POST `
        $AnnotatedUrl `
        -F "file=@$InputImage" `
        --output $EvidencePng

    Assert-LastExitCode "curl endpoint PNG"

    $Png = Get-Item $EvidencePng

    if ($Png.Length -le 8) {
        throw "O endpoint PNG retornou um arquivo vazio/inválido."
    }

    $Bytes = [System.IO.File]::ReadAllBytes($Png.FullName)

    # Assinatura oficial de um arquivo PNG:
    # 89 50 4E 47 0D 0A 1A 0A
    $ExpectedSignature = @(137, 80, 78, 71, 13, 10, 26, 10)

    for ($Index = 0; $Index -lt 8; $Index++) {
        if ($Bytes[$Index] -ne $ExpectedSignature[$Index]) {
            throw "O arquivo retornado não possui assinatura PNG válida."
        }
    }

    Write-Host "PNG OK -> $EvidencePng ($($Png.Length) bytes)"

    # =============================================================
    # 7. BUILD MULTIARQUITETURA
    # =============================================================
    if (-not $SkipMultiArch) {
        Write-Step "7/7 - Build multiarch amd64 + arm64"

        docker buildx build `
            --platform linux/amd64,linux/arm64 `
            --tag person-detected:multiarch `
            --output=type=cacheonly `
            .

        Assert-LastExitCode "docker buildx build multiarch"
    }
    else {
        Write-Step "7/7 - Build multiarch pulado por opção"
    }

    # =============================================================
    # RESULTADO FINAL
    # =============================================================
    Write-Step "PARTE 1 - VALIDAÇÃO CONCLUÍDA"

    docker compose ps

    Write-Host ""
    Write-Host "Evidências locais:"
    Write-Host "  JSON: $EvidenceJson"
    Write-Host "  PNG : $EvidencePng"
    Write-Host ""
    Write-Host "Swagger: http://127.0.0.1:$Port/docs"
}
finally {
    if (-not $KeepRunning) {
        Write-Host ""
        Write-Host "Encerrando Docker Compose..."
        docker compose down
    }
    else {
        Write-Host ""
        Write-Host "Containers mantidos ativos (-KeepRunning)."
    }
}
