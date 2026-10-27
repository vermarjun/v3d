#!/bin/sh
# Download the Brush (Gaussian-splat trainer, Rust/wgpu) macOS arm64 release into tools/brush/.
set -e
VER=${BRUSH_VERSION:-v0.3.0}
DIR="$(cd "$(dirname "$0")" && pwd)/brush"
mkdir -p "$DIR"
curl -fL "https://github.com/ArthurBrussee/brush/releases/download/$VER/brush-app-aarch64-apple-darwin.tar.xz" \
  | tar -xJ -C "$DIR" --strip-components=1
xattr -d com.apple.quarantine "$DIR/brush_app" 2>/dev/null || true
echo "installed $DIR/brush_app"
