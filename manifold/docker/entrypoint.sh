#!/usr/bin/env bash
# ================================================================
# Dual-mode entrypoint for RunAI containers
#
# No arguments  -> Interactive mode (SSH daemon for development)
# With arguments -> Job mode (activate venv, exec command)
# ================================================================
set -euo pipefail

APP_USER="${APP_USER:-appuser}"
APP_GID="${APP_GID:-100}"

# ---- Generate SSH host keys on first boot ----
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -A >/dev/null 2>&1
fi

# ---- Set up authorized_keys from persistent storage ----
# Install the same key for both APP_USER and root (root ssh is allowed by the
# image's sshd_config), so `ssh appuser@` and `ssh root@` both work by key.
install_authorized_keys() {
    ssh_dir="$1"; owner="$2"
    mkdir -p "$ssh_dir"
    if [ -f "/myhome/.ssh/authorized_keys" ]; then
        cp /myhome/.ssh/authorized_keys "$ssh_dir/authorized_keys"
    elif [ -f "/myhome/.ssh/id_ed25519.pub" ]; then
        cp /myhome/.ssh/id_ed25519.pub "$ssh_dir/authorized_keys"
    elif [ -f "/myhome/.ssh/id_rsa.pub" ]; then
        cp /myhome/.ssh/id_rsa.pub "$ssh_dir/authorized_keys"
    fi
    chmod 700 "$ssh_dir"
    [ -f "$ssh_dir/authorized_keys" ] && chmod 600 "$ssh_dir/authorized_keys"
    chown -R "$owner" "$ssh_dir"
}
install_authorized_keys "/home/${APP_USER}/.ssh" "${APP_USER}:${APP_GID}"
install_authorized_keys "/root/.ssh" "root:root"

# ---- Faiss ----
# faiss is now compiled and installed at image build time (see Dockerfile);
# no per-boot compilation needed.

# ---- Maldi Repo Check and Install ----
if [ ! -d /myhome/mlibra ]; then
echo "[entrypoint] recreating /mydata/maldi. Cloning..."
    gosu "${APP_USER}" git clone https://github.com/SwissDataScienceCenter/mlibra.git /myhome/mlibra
fi
if [! -d /myhome/mlibra/euclid]; then
echo "[entrypoint] recreating /mydata/maldi/euclid. Cloning..."
    gosu git clone https://github.com/lamanno-epfl/EUCLID.git /myhome/mlibra/euclid
fi
cd /myhome/mlibra

# Editable install into the venv
# We do this every boot to ensure the venv recognizes the local folder
echo "[entrypoint] Installing maldi in editable mode..."
pip install -e /myhome/mlibra

if [ $# -eq 0 ]; then
    # ============================================================
    # Interactive mode — start sshd for remote development
    # ============================================================

    # VS Code server + extensions (first boot only)
    vscode_dir="/home/${APP_USER}/.vscode-server"
    if [ -f "/myhome/vscode-extensions.txt" ] && [ ! -f "$vscode_dir/.extensions-installed" ]; then
        echo "[entrypoint] Installing VS Code extensions (first boot)..."
        mkdir -p "$vscode_dir"

        VSCODE_ARCH="$(uname -m)"
        case "${VSCODE_ARCH}" in
            x86_64) VSCODE_ARCH="x64" ;;
            aarch64|arm64) VSCODE_ARCH="arm64" ;;
        esac

        # Download VS Code server if not already present
        if ! find "$vscode_dir"/bin -name "code-server" -type f 2>/dev/null | grep -q .; then
            DOWNLOAD_URL="https://update.code.visualstudio.com/latest/server-linux-${VSCODE_ARCH}/stable"
            curl -fsSL "$DOWNLOAD_URL" -o "/tmp/vscode-server.tar.gz"
            mkdir -p "$vscode_dir/bin/stable"
            tar --strip-components=1 -xzf "/tmp/vscode-server.tar.gz" -C "$vscode_dir/bin/stable"
            rm "/tmp/vscode-server.tar.gz"
        fi

        # Install extensions from the user's list
        VSCODE_BIN=$(find "$vscode_dir"/bin -name "code-server" -type f 2>/dev/null | head -1)
        if [ -n "$VSCODE_BIN" ]; then
            while IFS= read -r ext || [ -n "$ext" ]; do
                [ -z "$ext" ] || [ "${ext:0:1}" = "#" ] && continue
                echo "  Installing: $ext"
                "$VSCODE_BIN" --install-extension "$ext" 2>/dev/null || true
            done < /myhome/vscode-extensions.txt
        fi

        touch "$vscode_dir/.extensions-installed"
        chown -R "${APP_USER}:${APP_GID}" "$vscode_dir"
    fi

    echo "[entrypoint] Starting sshd on port 2222 (interactive mode)"
    exec /usr/sbin/sshd -D -e

else
    # ============================================================
    # Job mode — activate venv and exec the provided command
    # ============================================================
    # gosu drops privileges from root to APP_USER for proper
    # PID 1 signal handling (unlike su, gosu execs directly).
    exec gosu "${APP_USER}" bash -c 'source /opt/venv/bin/activate && exec "$@"' -- "$@"
fi
