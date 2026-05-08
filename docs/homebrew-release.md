# Homebrew Release Plan

Dictate should support Homebrew, but only after release artifacts are reliable.
The cask in `Cask/dictate.rb` is a template until a real DMG and SHA are
published for the current version.

## Target User Flow

```bash
brew tap 0xbrando/dictate
brew install --cask dictate
```

## Release Checklist

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
   brew tap 0xbrando/dictate
   brew install --cask dictate
   ```

## Notes

- Keep `pip install dictate-mlx` as the primary install path until the cask has a
  real release artifact.
- The cask should install the full app experience, not a degraded CLI-only build.
- The Swift ANE helper should be bundled or discoverable by the app before Brew is
  promoted as the recommended install path.
- Notarization is not strictly required for an early technical release, but it is
  the right bar before pitching Dictate to non-developer Mac users.
