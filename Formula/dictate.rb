class Dictate < Formula
  include Language::Python::Virtualenv

  desc "Local push-to-talk dictation for Apple Silicon Macs"
  homepage "https://github.com/0xbrando/dictate"
  url "https://github.com/0xbrando/dictate/archive/refs/tags/v2.5.2.tar.gz"
  sha256 "476c96603cf26541a16ff52fd4d00a7da0c28a760cd6c12fbf02ad86f85ce645"
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
      The selected local models download on first use and stay cached locally.
    EOS
  end
end
