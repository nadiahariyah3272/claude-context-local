param(
    [string]$RepoUrl = "https://github.com/FarhanAliRaza/claude-context-local",
    [string]$ProjectDir = "$env:LOCALAPPDATA\claude-context-local",
    [string]$StorageDir = "$env:USERPROFILE\.claude_code_search",
    [string]$ModelName = $(if ($env:CODE_SEARCH_MODEL) { $env:CODE_SEARCH_MODEL } else { "google/embeddinggemma-300m" })
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "=================================================="
    Write-Host $Message
    Write-Host "=================================================="
    Write-Host ""
}

Write-Section "Installing Claude Context Local"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required. Please install git and re-run."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $uvBinary = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $uvBinary) {
        $env:PATH = "$($env:USERPROFILE)\.local\bin;$env:PATH"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation failed or was not added to PATH."
    }
}

$skipUpdate = $false
$isUpdate = $false

if (Test-Path (Join-Path $ProjectDir ".git")) {
    $isUpdate = $true
    Write-Host "Found existing installation at $ProjectDir"
    Push-Location $ProjectDir
    try {
        git diff-index --quiet HEAD --
        $hasChanges = $LASTEXITCODE -ne 0
    } finally {
        Pop-Location
    }

    if ($hasChanges) {
        Write-Warning "You have uncommitted changes in $ProjectDir"
        $choice = Read-Host "Options: [U]pdate anyway (stash changes), [K]eep current version, [D]elete and reinstall"
        if ([string]::IsNullOrWhiteSpace($choice)) {
            $choice = "U"
        }
        switch ($choice.ToUpperInvariant()) {
            "K" { $skipUpdate = $true }
            "D" {
                Remove-Item -Recurse -Force $ProjectDir
                git clone $RepoUrl $ProjectDir
                $isUpdate = $false
            }
            default {
                Push-Location $ProjectDir
                try {
                    git stash push -m "Auto-stash before installer update $(Get-Date -Format s)"
                    git remote set-url origin $RepoUrl
                    git fetch --tags --prune
                    git pull --ff-only
                } finally {
                    Pop-Location
                }
                Write-Host "Your changes are stashed. Run 'git stash pop' in $ProjectDir to restore them."
            }
        }
    }
    elseif (-not $skipUpdate) {
        Push-Location $ProjectDir
        try {
            git remote set-url origin $RepoUrl
            git fetch --tags --prune
            git pull --ff-only
        } finally {
            Pop-Location
        }
    }
}
else {
    $projectParent = Split-Path -Parent $ProjectDir
    if ($projectParent) {
        New-Item -ItemType Directory -Force -Path $projectParent | Out-Null
    }
    if (Test-Path $ProjectDir) {
        Remove-Item -Recurse -Force $ProjectDir
    }
    git clone $RepoUrl $ProjectDir
}

if (-not $skipUpdate) {
    Write-Host "Installing Python dependencies with uv"
    Push-Location $ProjectDir
    try {
        uv sync
    } finally {
        Pop-Location
    }
}
else {
    Write-Host "Skipping dependency update (keeping current version)"
}

$skipGpu = $env:SKIP_GPU -eq "1"
if (-not $skipGpu) {
    Write-Host "Checking for NVIDIA GPU to install FAISS GPU wheels (optional)"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "NVIDIA GPU detected. Attempting to install faiss-gpu wheels..."
        Push-Location $ProjectDir
        try {
            uv add faiss-gpu-cu12 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Installed faiss-gpu-cu12"
                uv remove faiss-cpu | Out-Null
            }
            else {
                uv add faiss-gpu-cu11 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "Installed faiss-gpu-cu11"
                    uv remove faiss-cpu | Out-Null
                }
                else {
                    Write-Host "Could not install faiss-gpu wheels. Keeping faiss-cpu."
                }
            }
        } finally {
            Pop-Location
        }
    }
    else {
        Write-Host "No NVIDIA GPU detected. Using faiss-cpu (default)."
    }
}

Write-Host "Downloading embedding model to $StorageDir"
New-Item -ItemType Directory -Force -Path $StorageDir | Out-Null
$downloadSucceeded = $true
Push-Location $ProjectDir
try {
    uv run scripts/download_model_standalone.py --storage-dir $StorageDir --model $ModelName -v
    if ($LASTEXITCODE -ne 0) {
        $downloadSucceeded = $false
    }
} finally {
    Pop-Location
}

if (-not $downloadSucceeded) {
    Write-Warning "Model download did not complete. The install still succeeded, but indexing/search will need Hugging Face access before first use."
}

Write-Section ($(if ($isUpdate) { "Update complete!" } else { "Install complete!" }))
Write-Host "Project: $ProjectDir"
Write-Host "Storage: $StorageDir"
Write-Host "Selected embedding model: $ModelName"
Write-Host "Local install model config: $(Join-Path $StorageDir 'install_config.json')"
Write-Host ""
Write-Host "Next steps:"
if ($isUpdate) {
    Write-Host "1) Remove old server: claude mcp remove code-search"
    Write-Host "2) Add updated server: claude mcp add code-search --scope user -- uv run --directory `"$ProjectDir`" python mcp_server/server.py"
}
else {
    Write-Host "1) Add MCP server: claude mcp add code-search --scope user -- uv run --directory `"$ProjectDir`" python mcp_server/server.py"
}
Write-Host "2) Verify connection: claude mcp list"
Write-Host "3) In Claude Code: index this codebase"
Write-Host "4) To switch embedding models later, set CODE_SEARCH_MODEL and re-run this installer"
