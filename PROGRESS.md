# BirdLens progress log

## ACTION NEEDED

**Reason:** Two things on this PC need admin rights, a reboot and some GUI clicks, so I can't do them myself. WSL isn't installed (`wsl` fails with `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED`), so training can't start. Android Studio is installed but its first-run setup was never completed, so there's no Android SDK and no emulator (`flutter doctor`: "Unable to locate Android SDK").

**What I need you to do (about 20 to 30 minutes, mostly downloads):**

1. Install WSL2 with Ubuntu. Open PowerShell as Administrator and run:
   ```
   wsl --install -d Ubuntu-24.04
   ```
   Reboot when it asks. After the reboot, open "Ubuntu" from the Start menu once and create a Linux username and password. Remember the password: I'll never ask for it, but you may need it if a step needs `sudo`.
2. Finish the Android Studio setup. Open Android Studio, go through the setup wizard with "Standard" install type, and let it download the SDK. Then open More Actions > Virtual Device Manager, create a Pixel device (for example Pixel 8) with a recent x86_64 system image, and start it once to check it boots.
3. In a normal PowerShell window, run `flutter doctor --android-licenses` and accept the licences (type `y`).
4. Come back to this Claude session and say "continue".

Once those are done I can carry on through Phases 2 to 8 without stopping, unless TensorFlow can't use your GPU (see the note under Phase 0).

---

## Phase 0: Environment check (2026-10-23)

**GPU** (from `nvidia-smi` on Windows; I couldn't run it inside WSL yet):
- NVIDIA GeForce RTX 5060, 8151 MiB VRAM, driver 591.44
- Plan: cap TensorFlow at about 4000 MiB (about 50 percent) with memory growth on.
- Risk to check: the RTX 5060 is a Blackwell card (compute capability 12.0). Once WSL is up, I'll check whether the pinned TensorFlow build has kernels for it or can JIT from PTX. If it can't, I'll report it rather than silently falling back to CPU.

**Other machine facts:**
- RAM about 16 GB, about 36 GB free on C:. Expected disk use: about 6 GB for training, plus about 8 GB for the Android SDK and emulator image, which fits.
- Flutter 3.47.7 stable, Dart 3.13.5. `flutter doctor -v`: Flutter OK. Android toolchain missing (needs the Studio wizard, see above). Visual Studio missing, which only matters for Windows desktop builds, so I'm ignoring it.
- Java on PATH is Oracle Java 8. Gradle builds will use Android Studio's bundled JBR instead once the SDK is set up.
- git 2.56.0, author already set to Lucas Lee <lucalele1007@gmail.com>.

**Decisions from the Q&A before starting:**
- Repo lives at `C:\Users\Admin\birdlens`. WSL sees it at `/mnt/c/Users/Admin/birdlens`. The heavy files (dataset, venv, checkpoints) go in the WSL home folder for speed and are linked into `training/`. They're gitignored either way.
- Commits are authored as Lucas Lee with no AI co-author trailer.
- The emulator can be started at any time during Phase 7.
- The benchmark label reads "Android emulator on my desktop PC, not a real phone".
- Ubuntu 24.04 ships Python 3.12, so I'll get Python 3.11 through `uv` without needing sudo.

## Phase 1: Repo setup (2026-10-23)

- Created the folder layout: `training/`, `training/logs/`, `app/`, `docs/`, `docs/screenshots/`.
- Added `.gitignore` covering the dataset, venv, checkpoints, logs, intermediate exports and Flutter/Android build outputs. `.tflite` files are ignored everywhere except `app/assets/models/`.
- Added an MIT `LICENSE` for my code, with a note that the CUB images are not covered, plus a stub README.
- Ran `git init` and made the first commit. No remote added.
- Next: Phase 2, once WSL is installed.
