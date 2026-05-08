# Homebrew Cask for Dictate
# Install: brew tap 0xbrando/dictate && brew install --cask dictate
# Or submit to homebrew/cask once notable enough

cask "dictate" do
  version "2.5.2"
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

    Dictate downloads only the selected local models on first use, then caches
    them in ~/.cache/huggingface. Other models are optional one-click downloads.

    All processing happens locally on your Mac; nothing leaves your device unless
    you explicitly opt into a remote API endpoint.
  EOS
end
