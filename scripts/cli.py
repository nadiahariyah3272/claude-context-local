#!/usr/bin/env python3
"""Command-line interface for Claude Context Local.

Provides help, diagnostics, and management commands for the local
semantic code search system.
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common_utils import VERSION, get_storage_dir, load_local_install_config

# ── Colour helpers (degrade gracefully when stdout is not a terminal) ──

_NO_COLOR = os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty()


def _clr(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _clr("1", text)


def green(text: str) -> str:
    return _clr("32", text)


def yellow(text: str) -> str:
    return _clr("33", text)


def red(text: str) -> str:
    return _clr("31", text)


def cyan(text: str) -> str:
    return _clr("36", text)


def _get_storage_dir_or_report(command_name: str) -> Optional[Path]:
    """Return the storage directory or print actionable guidance."""
    try:
        return get_storage_dir()
    except RuntimeError as exc:
        print(f"{red('✗')} {command_name} could not access the storage directory.")
        print(f"  {exc}")
        if "CODE_SEARCH_STORAGE" not in str(exc):
            print(f"  Set {cyan('CODE_SEARCH_STORAGE')} to a writable path and try again.")
        print()
        return None


# ── Platform helpers ──────────────────────────────────────────────────

def is_windows() -> bool:
    return platform.system() == "Windows"


def is_wsl() -> bool:
    """Detect if running inside WSL."""
    if platform.system() != "Linux":
        return False
    try:
        release = Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
        return "microsoft" in release or "wsl" in release
    except OSError:
        return False


def get_platform_label() -> str:
    system = platform.system()
    if is_wsl():
        return "WSL2 (Windows Subsystem for Linux)"
    return {"Windows": "Windows", "Darwin": "macOS", "Linux": "Linux"}.get(system, system)


def get_default_install_dir() -> Path:
    """Return the expected installation directory for this platform."""
    if is_windows():
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            return Path(local_app) / "claude-context-local"
        return Path.home() / "AppData" / "Local" / "claude-context-local"
    return Path.home() / ".local" / "share" / "claude-context-local"


def get_claude_config_paths() -> list:
    """Return likely Claude configuration file paths for this platform."""
    paths = []
    home = Path.home()

    if is_windows():
        # Windows: AppData paths
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            paths.append(Path(localappdata) / "Claude" / "claude_desktop_config.json")
        paths.append(home / ".claude.json")
    elif platform.system() == "Darwin":
        paths.append(home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
        paths.append(home / ".claude.json")
    else:
        # Linux/WSL
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))
        paths.append(Path(xdg_config) / "Claude" / "claude_desktop_config.json")
        paths.append(home / ".claude.json")

    if is_wsl():
        # Also check Windows-side paths from WSL
        for win_user_dir in _wsl_windows_user_dirs():
            paths.append(win_user_dir / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json")

    return paths


def _wsl_windows_user_dirs() -> list:
    """Attempt to find Windows user directories from within WSL."""
    dirs = []
    # /mnt/c/Users/<username> is the typical WSL mount
    users_dir = Path("/mnt/c/Users")
    if users_dir.is_dir():
        for entry in users_dir.iterdir():
            if entry.is_dir() and entry.name not in ("Public", "Default", "Default User", "All Users"):
                dirs.append(entry)
    return dirs


# ── Sub-commands ──────────────────────────────────────────────────────

def cmd_help() -> None:
    """Print the main help message."""
    install_dir = get_default_install_dir()

    print(bold("Claude Context Local") + f"  v{VERSION}")
    print("Local semantic code search for Claude Code via MCP.\n")

    print(bold("USAGE"))
    print(f"  python scripts/cli.py {cyan('<command>')}\n")

    print(bold("COMMANDS"))
    cmds = [
        ("help", "Show this help message"),
        ("doctor", "Check installation health and diagnose problems"),
        ("version", "Print version and platform info"),
        ("status", "Show index statistics and active project info"),
        ("paths", "Show all paths used by the tool"),
        ("setup-guide", "Print step-by-step setup instructions for your OS"),
    ]
    for name, desc in cmds:
        print(f"  {cyan(name):<20s} {desc}")

    print(f"\n{bold('EXAMPLES')}")
    print(f"  python scripts/cli.py doctor        # Verify installation")
    print(f"  python scripts/cli.py setup-guide    # Setup instructions")
    print(f"  python scripts/cli.py status         # Show project status\n")

    print(f"{bold('MCP SERVER')}")
    if is_windows():
        print(f"  uv run --directory \"{install_dir}\" python mcp_server/server.py")
    else:
        print(f"  uv run --directory {install_dir} python mcp_server/server.py")

    print(f"\nSee README.md for full documentation.")


def cmd_version() -> None:
    """Print version and platform information."""
    print(f"claude-context-local  {VERSION}")
    print(f"Platform:  {get_platform_label()} ({platform.machine()})")
    print(f"Python:    {platform.python_version()}")


def cmd_paths() -> None:
    """Show all paths used by the tool."""
    install_dir = get_default_install_dir()
    print(bold("Paths used by Claude Context Local\n"))
    storage = _get_storage_dir_or_report("paths")

    if storage is None:
        print(f"  {yellow('—')} Install directory:       {install_dir}")
        print(f"\n{bold('Claude config locations (checked in order):')}")
        for p in get_claude_config_paths():
            marker = green("✓") if p.is_file() else yellow("—")
            print(f"  {marker} {p}")
        return

    rows = [
        ("Storage directory", str(storage), storage.is_dir()),
        ("Install directory", str(install_dir), install_dir.is_dir()),
        ("Models cache", str(storage / "models"), (storage / "models").is_dir()),
        ("Install config", str(storage / "install_config.json"), (storage / "install_config.json").is_file()),
        ("Projects data", str(storage / "projects"), (storage / "projects").is_dir()),
    ]

    for label, path, exists in rows:
        marker = green("✓") if exists else yellow("—")
        print(f"  {marker} {label + ':':<22s} {path}")

    print(f"\n{bold('Claude config locations (checked in order):')}")
    for p in get_claude_config_paths():
        marker = green("✓") if p.is_file() else yellow("—")
        print(f"  {marker} {p}")


def cmd_doctor() -> None:
    """Run diagnostic checks and report problems."""
    print(bold("Running diagnostics…\n"))
    issues = []
    storage = _get_storage_dir_or_report("doctor")

    # 1. Python version
    py = sys.version_info
    if py >= (3, 12):
        print(f"  {green('✓')} Python {py.major}.{py.minor}.{py.micro}")
    else:
        msg = f"Python >= 3.12 required (found {py.major}.{py.minor}.{py.micro})"
        print(f"  {red('✗')} {msg}")
        issues.append(msg)

    # 2. uv available
    if shutil.which("uv"):
        print(f"  {green('✓')} uv is installed")
    else:
        msg = "uv not found in PATH – install from https://astral.sh/uv/"
        print(f"  {red('✗')} {msg}")
        issues.append(msg)

    # 3. git available
    if shutil.which("git"):
        print(f"  {green('✓')} git is installed")
    else:
        msg = "git not found in PATH"
        print(f"  {red('✗')} {msg}")
        issues.append(msg)

    # 4. Storage directory
    if storage and storage.is_dir():
        print(f"  {green('✓')} Storage directory exists: {storage}")
    else:
        msg = "Storage directory unavailable – set CODE_SEARCH_STORAGE to a writable path"
        print(f"  {red('✗')} {msg}")
        issues.append(msg)

    # 5. Install config
    if storage:
        config = load_local_install_config(storage)
        if config:
            model = config.get("embedding_model", {})
            model_name = model.get("model_name", "unknown") if isinstance(model, dict) else (model or "unknown")
            print(f"  {green('✓')} Install config found (model: {model_name})")
        else:
            msg = "No install_config.json found – run the installer first"
            print(f"  {yellow('!')} {msg}")
            issues.append(msg)
    else:
        msg = "Install config unavailable because the storage directory is not writable"
        print(f"  {yellow('!')} {msg}")
        issues.append(msg)

    # 6. Models directory
    if storage:
        models_dir = storage / "models"
        if models_dir.is_dir() and any(models_dir.iterdir()):
            print(f"  {green('✓')} Models cached in: {models_dir}")
        else:
            msg = "No models cached yet – the embedding model needs to be downloaded"
            print(f"  {yellow('!')} {msg}")
            issues.append(msg)
    else:
        msg = "Model cache unavailable because the storage directory is not writable"
        print(f"  {yellow('!')} {msg}")
        issues.append(msg)

    # 7. Key Python packages
    for pkg_name, import_name in [
        ("faiss-cpu", "faiss"),
        ("sentence-transformers", "sentence_transformers"),
        ("fastmcp", "mcp.server.fastmcp"),
        ("tree-sitter", "tree_sitter"),
    ]:
        try:
            __import__(import_name)
            print(f"  {green('✓')} {pkg_name} importable")
        except Exception as exc:
            msg = f"{pkg_name} not importable ({type(exc).__name__}) – run 'uv sync' to install dependencies"
            print(f"  {red('✗')} {msg}")
            issues.append(msg)

    # 8. Claude CLI
    if shutil.which("claude"):
        print(f"  {green('✓')} Claude CLI found in PATH")
    else:
        msg = "Claude CLI not found in PATH – install from https://claude.ai/code"
        print(f"  {yellow('!')} {msg}")
        issues.append(msg)

    # 9. WSL-specific checks
    if is_wsl():
        print(f"\n  {cyan('ℹ')} WSL2 detected – checking Windows interop…")
        win_dirs = _wsl_windows_user_dirs()
        if win_dirs:
            print(f"  {green('✓')} Windows user directories accessible: {', '.join(d.name for d in win_dirs)}")
        else:
            msg = "Cannot access Windows user directories from WSL (/mnt/c/Users/)"
            print(f"  {yellow('!')} {msg}")
            issues.append(msg)

    # Summary
    print()
    if not issues:
        print(green(bold("All checks passed!")))
    else:
        print(yellow(bold(f"{len(issues)} issue(s) found:")))
        for issue in issues:
            print(f"  • {issue}")
        print(f"\nRun '{cyan('python scripts/cli.py setup-guide')}' for setup instructions.")


def cmd_status() -> None:
    """Show index statistics and active project info."""
    print(bold("Index Status\n"))
    storage = _get_storage_dir_or_report("status")
    if storage is None:
        return

    projects_dir = storage / "projects"

    if not projects_dir.is_dir():
        print("  No projects indexed yet.")
        print(f"  Use Claude Code to say: {cyan('index this codebase')}")
        return

    project_count = 0
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue

        info_file = project_dir / "project_info.json"
        stats_file = project_dir / "index" / "stats.json"

        if not info_file.is_file():
            continue

        project_count += 1
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info = {}

        name = info.get("project_name", project_dir.name)
        path = info.get("project_path", "unknown")
        print(f"  {bold(name)}")
        print(f"    Path: {path}")

        if stats_file.is_file():
            try:
                stats = json.loads(stats_file.read_text(encoding="utf-8"))
                chunks = stats.get("total_chunks", 0)
                files = stats.get("files_indexed", 0)
                print(f"    Chunks: {chunks}  Files: {files}")
            except (json.JSONDecodeError, OSError):
                print("    (stats unavailable)")
        else:
            print("    (not yet indexed)")

        print()

    if project_count == 0:
        print("  No projects indexed yet.")


def cmd_setup_guide() -> None:
    """Print step-by-step setup instructions for the current platform."""
    plat = get_platform_label()
    install_dir = get_default_install_dir()

    print(bold(f"Setup Guide for {plat}\n"))

    # Step 1 – Install
    print(bold("1. Install"))
    if is_windows():
        print("   Open PowerShell and run:\n")
        print(f"   {cyan('irm https://raw.githubusercontent.com/nadiahariyah3272/claude-context-local/main/scripts/install.ps1 | iex')}\n")
        print("   If execution policy blocks the script:\n")
        print(f"   {cyan('powershell -ExecutionPolicy Bypass -c \"irm https://raw.githubusercontent.com/nadiahariyah3272/claude-context-local/main/scripts/install.ps1 | iex\"')}\n")
    elif is_wsl():
        print("   From your WSL terminal:\n")
        print(f"   {cyan('curl -fsSL https://raw.githubusercontent.com/nadiahariyah3272/claude-context-local/main/scripts/install.sh | bash')}\n")
        print(f"   {yellow('Note:')} If Claude Desktop is installed on the Windows side,")
        print(f"   you may need to register the MCP server using the Windows path.")
        print(f"   The installer puts the project at: {install_dir}\n")
    else:
        print("   In your terminal:\n")
        print(f"   {cyan('curl -fsSL https://raw.githubusercontent.com/nadiahariyah3272/claude-context-local/main/scripts/install.sh | bash')}\n")

    # Step 2 – Register MCP
    print(bold("2. Register the MCP server"))
    if is_windows():
        print(f"   {cyan(f'claude mcp add code-search --scope user -- uv run --directory \"{install_dir}\" python mcp_server/server.py')}\n")
    else:
        print(f"   {cyan(f'claude mcp add code-search --scope user -- uv run --directory {install_dir} python mcp_server/server.py')}\n")

    if is_wsl():
        print(f"   {yellow('WSL tip:')} If Claude Desktop runs on Windows, register the server")
        print(f"   from a Windows terminal using the Windows-style path.\n")

    # Step 3 – Verify
    print(bold("3. Verify"))
    print(f"   {cyan('claude mcp list')}")
    print(f"   Look for: code-search … {green('✓ Connected')}\n")

    # Step 4 – Use
    print(bold("4. Index & search"))
    print(f"   Open Claude Code in your project directory and say:")
    print(f"   {cyan('index this codebase')}\n")
    print(f"   Then search with:")
    print(f"   {cyan('search for authentication logic')}\n")

    # Troubleshooting
    print(bold("Troubleshooting"))
    print(f"  • Run {cyan('python scripts/cli.py doctor')} to diagnose issues")
    print(f"  • If model download fails, authenticate: {cyan('hf auth login')}")
    if is_windows():
        print(f"  • On Windows, set HF_TOKEN in the same shell: {cyan('$env:HF_TOKEN=\"hf_xxx\"')}")
    if is_wsl():
        print(f"  • In WSL, set HF_TOKEN explicitly: {cyan('export HF_TOKEN=hf_xxx')}")
    print(f"  • Ensure Python >= 3.12 and uv are installed")
    print()


# ── Entry point ───────────────────────────────────────────────────────

COMMANDS = {
    "help": cmd_help,
    "--help": cmd_help,
    "-h": cmd_help,
    "doctor": cmd_doctor,
    "version": cmd_version,
    "--version": cmd_version,
    "status": cmd_status,
    "paths": cmd_paths,
    "setup-guide": cmd_setup_guide,
}


def main() -> None:
    args = sys.argv[1:]
    if not args:
        cmd_help()
        return

    command = args[0].lower()
    handler = COMMANDS.get(command)
    if handler is None:
        print(red(f"Unknown command: '{command}'"))
        print(f"Run '{cyan('python scripts/cli.py help')}' to see available commands.\n")
        sys.exit(1)

    handler()


if __name__ == "__main__":
    main()
