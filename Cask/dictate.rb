# Homebrew Cask for Dictate
# Install: brew tap 0xbrando/dictate && brew install --cask dictate
# Or submit to homebrew/cask once notable enough

cask "dictate" do
  version "2.4.1"
  sha256 "PLACEHOLDER_UPDATE_ON_RELEASE"

  url "https://github.com/0xbrando/dictate/releases/download/v#{version}/Dictate-v#{version}-arm64.dmg"
  name "Dictate"
  desc "Push-to-talk voice dictation, 100% local on Apple Silicon"
  homepage "https://github.com/0xbrando/dictate"

  depends_on macos: ">= :sonoma"
  depends_on arch: :arm64

  app "Dictate.app"

  zap trash: [
    "~/Library/Application Support/Dictate",
    "~/Library/Logs/Dictate",
    "~/Library/Preferences/com.0xbrando.dictate.plist",
  ]

  caveats <<~EOS
    Dictate requires microphone and accessibility permissions.
    Grant them in System Settings → Privacy & Security when prompted.

    Speech models are downloaded on first launch (~75 MB for default model).
    All processing happens locally on your Mac — nothing leaves your device.
  EOS
end
