# Dictate Launch Checklist

Positioning: **"SuperWhisper but free and open source"**
Target: macOS developers, privacy-conscious users, AI builders

---

## Phase 0 — Pre-Launch (Before Going Public)

### README Polish
- [x] 1-sentence pitch in first line ✅
- [x] Clear install command (`pip install dictate-mlx`) ✅
- [x] Feature table ✅
- [x] Environment variables documented ✅
- [ ] Add shields.io badges (license, Python version, tests passing, coverage %)
- [ ] Add hero GIF: 10-second screen recording showing hold-key → speak → text appears
- [ ] Add comparison table vs SuperWhisper/Wispr Flow/VoiceInk (free vs $8-39)
- [ ] Add "Why Dictate?" section (local, free, no account, no cloud, OSS)

### Code Quality
- [x] 828 tests, 97% coverage ✅
- [x] Security audit: clean ✅
- [x] CI: tests pass ✅
- [ ] Set up GitHub Actions CI (pytest + coverage badge)
- [ ] Add `CONTRIBUTING.md`
- [ ] Tag "good first issue" on 3-5 issues (attracts contributors)
- [ ] Public roadmap (GitHub Projects board, even minimal)

### Distribution
- [ ] **Homebrew cask** — requires .app bundle:
  1. Build with py2app → Dictate.app
  2. Code-sign with Developer ID (or ad-hoc for OSS)
  3. Create .dmg installer
  4. Upload to GitHub Releases
  5. Write Homebrew cask formula
  6. Submit PR to homebrew-cask
- [ ] **PyPI** — `pip install dictate-mlx` (already configured)
- [ ] **GitHub Releases** — tag v2.4.1, attach .dmg + source

---

## Phase 1 — Launch Week

### Day 1: Soft Launch
- [ ] Tweet from @0xBrando: demo thread (4-5 tweets)
  - Tweet 1: 10s GIF of dictation in action
  - Tweet 2: "100% local, no cloud, no API keys"
  - Tweet 3: Feature highlights (dual STT, translation, 12 languages)
  - Tweet 4: "828 tests, MIT license, made for developers"
  - Tweet 5: GitHub link + "star if useful"
- [ ] Post to r/macapps ("I built a free, open-source voice dictation app for macOS")
- [ ] Post to r/commandline

### Day 2: Hacker News
- [ ] "Show HN: Dictate – Push-to-talk voice dictation for macOS, 100% local (MLX)"
  - Lead with: "No cloud, no API keys, no subscriptions. Hold a key, speak, release — clean text appears."
  - Technical depth: dual STT engines, MLX acceleration, LLM cleanup
  - Be in comments for first 2 hours responding to everything
- [ ] Cross-post Show HN link to Twitter

### Day 3: ProductHunt
- [ ] Create maker profile for @0xBrando
- [ ] Prepare PH listing: tagline, screenshots, description
- [ ] Schedule for Tuesday morning (best day for PH)
- [ ] Share PH link across social

### Ongoing
- [ ] Submit to TLDR newsletter, Morning Dev, Changelog
- [ ] Dev.to post: "I built SuperWhisper's open-source alternative"
- [ ] Respond to ALL issues/feedback within 24h
- [ ] Weekly releases for first month

---

## Phase 2 — Sustain (First Month)

- [ ] Monitor GitHub stars trajectory (target: 200/day for Trending)
- [ ] Create Discord server for Dictate/Hawkeye/Sensei
- [ ] Add integrations section (Raycast, Alfred, Hammerspoon)
- [ ] Blog post: "How Dictate Uses MLX for Zero-Latency Transcription"
- [ ] Collect testimonials / user quotes for README

---

## Competitive Landscape (Feb 2026)

| Tool | Price | Local? | OSS? | Languages | Our Edge |
|------|-------|--------|------|-----------|----------|
| SuperWhisper | $8.49/mo | ✅ | ❌ | 99+ | Free, OSS, dual STT |
| Wispr Flow | $12/mo | ❌ | ❌ | 13 | 100% local, no cloud |
| VoiceInk | $39.99 | ✅ | ❌ | 100+ | Free, open-source |
| macOS Dictation | Free | ✅ | ❌ | ~40 | LLM cleanup, PTT, developer-focused |
| **Dictate** | **Free** | **✅** | **✅** | **12+** | **Only free + local + OSS + LLM cleanup** |

---

## Key Metrics to Track
- GitHub stars (target: 500 first week, 2K first month)
- Homebrew installs (via cask analytics)
- PyPI downloads
- Issues opened (engagement signal)
- Contributors (target: 5 in first month)

---

*Based on research: Ideas/oss-launch-strategy-feb2026.md*
*Created: 2026-02-12 night shift*
