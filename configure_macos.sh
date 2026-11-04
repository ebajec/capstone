#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

general() {
    if [ "$(uname -s)" != "Darwin" ]; then
        echo "This setup currently supports macOS only." >&2
        exit 1
    fi

    if ! xcrun --find clang++ >/dev/null 2>&1; then
        echo "Xcode Command Line Tools are required. Run xcode-select --install and re-run." >&2
        exit 1
    fi

    # Update git submodules from the repository root.
    cd "$SCRIPT_DIR"
    git submodule update --init --recursive
}

macos() {
    echo "Configuring macOS..."

    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew is required. Install from https://brew.sh and re-run."
        exit 1
    fi

    # Install dependencies
    DEPENDENCIES=(
        eigen
        libomp
        glfw
        glm
        shaderc
        cmake
        clang-format
        ninja
        yaml-cpp
        coreutils
    )

    brew install "${DEPENDENCIES[@]}"

    local sdk_setup="" candidate answer sdk_version download_dir installer
    # VULKAN_SDK normally points to the SDK's macOS subdirectory.
    if [ -n "${VULKAN_SDK:-}" ]; then
        for candidate in "$VULKAN_SDK/../setup-env.sh" "$VULKAN_SDK/setup-env.sh"; do
            if [ -f "$candidate" ]; then
                sdk_setup="$candidate"
                break
            fi
        done
    fi
    if [ -z "$sdk_setup" ]; then
        # Prefer the newest installed version in LunarG's default location.
        sdk_setup=$(find "$HOME/VulkanSDK" -mindepth 2 -maxdepth 2 -name setup-env.sh -type f 2>/dev/null |
            sort -V | tail -n 1) || true
    fi

    if [ -z "$sdk_setup" ]; then
        if ! read -r -p "VULKAN SDK NOT FOUND. DO YOU WANT TO INSTALL THE LATEST VULKAN SDK? [Y/N] " answer; then
            answer=n
        fi
        case "$answer" in
            [Yy]|[Yy][Ee][Ss]) ;;
            *) echo "Skipping Vulkan SDK installation. Vulkan is required to build this project."; return ;;
        esac

        sdk_version=$(curl -fsSL https://vulkan.lunarg.com/sdk/latest/mac.txt)
        if [[ ! "$sdk_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Could not determine the latest Vulkan SDK version." >&2
            return 1
        fi
        download_dir=$(mktemp -d)
        if ! curl -fL "https://sdk.lunarg.com/sdk/download/$sdk_version/mac/vulkan_sdk.zip" \
                -o "$download_dir/vulkan_sdk.zip" ||
                ! ditto -x -k "$download_dir/vulkan_sdk.zip" "$download_dir"; then
            rm -rf "$download_dir"
            return 1
        fi
        installer=$(find "$download_dir" -type f -path '*/Contents/MacOS/*' -iname 'vulkansdk-macos-*' -print -quit)
        if [ -z "$installer" ] || ! "$installer" --root "$HOME/VulkanSDK/$sdk_version" \
                --accept-licenses --default-answer --confirm-command install copy_only=1; then
            echo "Vulkan SDK installation failed." >&2
            rm -rf "$download_dir"
            return 1
        fi
        rm -rf "$download_dir"
        sdk_setup="$HOME/VulkanSDK/$sdk_version/setup-env.sh"
    fi

    if [ ! -f "$sdk_setup" ]; then
        echo "Vulkan SDK setup script not found: $sdk_setup" >&2
        return 1
    fi

    printf 'To enable Vulkan in your current terminal, run: source %q\n' "$sdk_setup"
}

# --- Main execution flow ---
general
macos

echo "Then configure and build:"
echo "  cmake -S . -B build -G Ninja"
echo "  ninja -C build"
