#!/bin/bash
# Build Dictate.app and optionally package as DMG
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
VERSION=$("$PYTHON" -c "from dictate import __version__; print(__version__)")
APP_NAME="Dictate"
DMG_NAME="${APP_NAME}-v${VERSION}-arm64.dmg"

echo "🔨 Building ${APP_NAME} v${VERSION}..."
echo

# Clean previous builds
rm -rf build dist

# Build standalone .app bundle from a temp directory so py2app doesn't inherit
# package metadata from pyproject.toml.
BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "$BUILD_ROOT"' EXIT
(cd "$BUILD_ROOT" && "$PYTHON" "$REPO_ROOT/setup_app.py" py2app --dist-dir "$REPO_ROOT/dist" --bdist-base "$BUILD_ROOT/build")

# Bundle the Swift ANE helper into Contents/MacOS so the app is self-contained.
SWIFT_HELPER="$REPO_ROOT/swift-stt/.build/release/dictate-stt"
if [[ ! -x "$SWIFT_HELPER" ]]; then
    echo "🔨 Building Swift ANE helper..."
    (cd "$REPO_ROOT/swift-stt" && swift build -c release)
fi
cp "$SWIFT_HELPER" "dist/${APP_NAME}.app/Contents/MacOS/dictate-stt"

# Ad-hoc code sign (free, allows right-click → Open)
echo
echo "🔏 Signing (ad-hoc)..."
codesign --deep --force --sign - "dist/${APP_NAME}.app"

echo "✅ Built: dist/${APP_NAME}.app"
echo

# Create DMG if create-dmg is available
if command -v create-dmg &>/dev/null; then
    echo "📦 Creating DMG..."
    create-dmg \
        --volname "${APP_NAME}" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "${APP_NAME}.app" 175 190 \
        --app-drop-link 425 190 \
        --hide-extension "${APP_NAME}.app" \
        "dist/${DMG_NAME}" \
        "dist/${APP_NAME}.app"
    echo "✅ Built: dist/${DMG_NAME}"
else
    # Fallback to zip
    echo "📦 Creating ZIP (install create-dmg for DMG output)..."
    (cd dist && zip -r "${APP_NAME}-v${VERSION}-arm64.zip" "${APP_NAME}.app")
    echo "✅ Built: dist/${APP_NAME}-v${VERSION}-arm64.zip"
    echo "💡 For DMG: brew install create-dmg"
fi

echo
echo "Done! Test your build:"
echo "  open dist/${APP_NAME}.app"
