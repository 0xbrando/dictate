# Homebrew Release Plan

Dictate supports two possible Homebrew paths:

- `Formula/dictate.rb`: source install that creates a Python virtualenv, builds
  the Swift ANE helper, and links `dictate` plus `dictate-stt`.
- Future cask: app-bundle install that needs a reliable DMG release artifact.
  Add `Cask/dictate.rb` only after packaging/notarization is ready.

## Target User Flow

```bash
brew tap 0xbrando/dictate
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
   brew tap 0xbrando/dictate
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
brew tap 0xbrando/dictate
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
