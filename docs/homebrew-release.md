# Homebrew Release Plan

## Publishing a version

1. Update the version in `pyproject.toml` and `dictate/__init__.py` together.
2. Run `python -m pytest tests/ -q`, then build with `python -m build` and
   validate both artifacts with `python -m twine check dist/*`.
3. Install the wheel in a fresh virtualenv, run `dictate --version` and
   `python -m pip check`, and check imports of the transcription and menu modules.
4. Merge the reviewed release commit. Tag that commit `vX.Y.Z`, upload the checked
   wheel and sdist to PyPI, and attach the same artifacts to a GitHub release.
   Keep publishing credentials outside the repository.
5. Download the new GitHub tag archive and compute its SHA-256. Update both the
   URL and checksum in `Formula/dictate.rb` through a follow-up PR.
6. Refresh the tap, build the formula and run `brew test 0xbrando/dictate/dictate`.
   Verify PyPI's published version and the tap's formula version before announcing.

The short tap command assumes a separate `homebrew-dictate` repository. This
project stores its formula in the main repository, so always include the explicit
repository URL below. The source formula requires Xcode 16+ (Swift 6), macOS 14+
and Apple Silicon. It resolves Python dependencies from PyPI inside an isolated
virtualenv; it is not an offline or fully locked Homebrew-core formula.

Dictate supports two possible Homebrew paths:

- `Formula/dictate.rb`: source install that creates a Python virtualenv, builds
  the Swift ANE helper, and links `dictate` plus `dictate-stt`.
- Future cask: app-bundle install that needs a reliable DMG release artifact.
  Add `Cask/dictate.rb` only after packaging/notarization is ready.

## Target User Flow

```bash
brew tap 0xbrando/dictate https://github.com/0xbrando/dictate
brew install dictate
```

## Formula Checklist

1. Tap the repo:

   ```bash
   brew tap 0xbrando/dictate https://github.com/0xbrando/dictate
   ```

2. Install the release formula:

   ```bash
   brew install --build-from-source 0xbrando/dictate/dictate
   ```

3. Run the formula test:

   ```bash
   brew test 0xbrando/dictate/dictate
   ```

4. Confirm both commands are available:

   ```bash
   dictate --version
   dictate-stt check
   ```

5. Advertise:

   ```bash
   brew tap 0xbrando/dictate https://github.com/0xbrando/dictate
   brew install dictate
   ```

## Future Cask Checklist

1. Build the app bundle:

   ```bash
   scripts/build-app.sh
   ```

2. Test the generated app locally:

   ```bash
   open dist/Dictate.app
   ```

3. Upload `dist/Dictate-vX.Y.Z-arm64.dmg` to the GitHub release.

4. Compute the cask SHA:

   ```bash
   shasum -a 256 dist/Dictate-vX.Y.Z-arm64.dmg
   ```

5. Update `Cask/dictate.rb` with the version and SHA.

6. Test the cask from the repo:

   ```bash
   brew install --cask ./Cask/dictate.rb
   ```

7. After the cask works locally, advertise:

   ```bash
brew tap 0xbrando/dictate https://github.com/0xbrando/dictate
brew install --cask dictate
   ```

## Notes

- The formula can be public before the cask because it builds from source and does
  not depend on a notarized DMG.
- The cask should install the full app experience, not a degraded CLI-only build.
- The Swift ANE helper should be bundled or discoverable by the app before Brew is
  promoted as the recommended install path.
- Notarization is not strictly required for an early technical release, but it is
  the right bar before pitching Dictate to non-developer Mac users.
