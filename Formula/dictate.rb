class Dictate < Formula
  include Language::Python::Virtualenv

  desc "Local push-to-talk dictation for Apple Silicon Macs"
  homepage "https://github.com/0xbrando/dictate"
  url "https://github.com/0xbrando/dictate/archive/refs/tags/v2.5.4.tar.gz"
  sha256 "74bffbf917c81fb04b85ae442c597de00bf989e287437dfd8bd595dbd7574979"
  license "MIT"

  head "https://github.com/0xbrando/dictate.git", branch: "main"

  depends_on "python@3.12"
  depends_on xcode: ["16.0", :build]
  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    system "swift", "build", "-c", "release", "--disable-sandbox", "--package-path", "swift-stt"
    bin.install "swift-stt/.build/release/dictate-stt"

    (libexec/"src").install "dictate", "pyproject.toml", "README.md", "LICENSE", "LICENSES.md"
    (bin/"dictate").write_env_script libexec/"venv/bin/dictate", {}
  end

  def post_install
    # Install upstream wheels after Homebrew's Mach-O relocation pass. Rewriting
    # their dylib IDs can fail (for example, tiktoken has no spare header space).
    venv = libexec/"venv"
    python = Formula["python@3.12"].opt_bin/"python3.12"
    virtualenv_create(venv, python, system_site_packages: false)
    # This third-party tap uses PyPI wheels for the ML stack. Homebrew's
    # pip_install_and_link passes --no-deps and leaves the app unable to start.
    system python, "-m", "pip", "--python=#{venv}/bin/python", "install",
           "--disable-pip-version-check", libexec/"src"
  end

  test do
    assert_match(/^dictate \d+\.\d+\.\d+$/, shell_output("#{bin}/dictate --version").strip)
    assert_match '"available":true', shell_output("#{bin}/dictate-stt check").delete(" ")
    system libexec/"venv/bin/python", "-c",
           "import mlx.core, mlx_whisper, mlx_lm, parakeet_mlx, sounddevice, scipy, pynput, pyperclip, rumps, dotenv"
  end

  def caveats
    <<~EOS
      Dictate runs as a macOS menu bar app:
        dictate

      Allow your terminal app in System Settings > Privacy & Security >
      Microphone and Accessibility, then quit and reopen Dictate.
      The selected local models download on first use and stay cached locally.
    EOS
  end
end
