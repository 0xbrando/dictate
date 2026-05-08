class Dictate < Formula
  include Language::Python::Virtualenv

  desc "Local push-to-talk dictation for Apple Silicon Macs"
  homepage "https://github.com/0xbrando/dictate"
  version "2.5.1"
  license "MIT"

  head "https://github.com/0xbrando/dictate.git", branch: "main"

  depends_on "python@3.11"
  depends_on xcode: ["15.0", :build]
  depends_on arch: :arm64

  def install
    system "swift", "build", "-c", "release", "--disable-sandbox", "--package-path", "swift-stt"
    bin.install "swift-stt/.build/release/dictate-stt"

    venv = virtualenv_create(libexec, "python3.11")
    venv.pip_install_and_link buildpath
  end

  test do
    assert_match(/^dictate \d+\.\d+\.\d+$/, shell_output("#{bin}/dictate --version").strip)
    assert_match '"available":true', shell_output("#{bin}/dictate-stt check").delete(" ")
  end

  def caveats
    <<~EOS
      Dictate runs as a macOS menu bar app:
        dictate

      macOS will ask for Microphone and Accessibility permissions.
      Models download lazily on first use and stay cached locally.
    EOS
  end
end
