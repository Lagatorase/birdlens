# BirdLens progress log

## ACTION NEEDED

**Reason:** CPU virtualization (AMD SVM) is turned off in the BIOS. Windows reports `VirtualizationFirmwareEnabled = False` for the Ryzen 5 5600G. Without it, WSL2 can't start any Linux distro: registering Ubuntu fails with error `0x80370102`, "enable the Virtual Machine Platform Windows feature and ensure virtualization is enabled in the BIOS". The Android emulator also has no hardware acceleration (`emulator -accel-check` says the hypervisor driver isn't installed), so it won't boot either. Changing the BIOS needs a reboot and GUI clicks, so I can't do it myself.

Already done: the Android SDK, the emulator `Medium_Phone_API_37.0` and the licences are all fine. WSL 3.0.1 and the Ubuntu app are installed, but the Ubuntu distro isn't registered yet.

**What I need you to do (about 10 minutes):**

1. Reboot into the BIOS. Press `Del` (or `F2`) repeatedly while the PC starts up.
2. Turn on **SVM Mode**. Where it lives depends on the board brand:
   - MSI: OC > Advanced CPU Configuration > SVM Mode
   - ASUS: Advanced > CPU Configuration > SVM Mode
   - Gigabyte: Tweaker > Advanced CPU Settings > SVM Mode
   - ASRock: Advanced > CPU Configuration > SVM Mode

   Set it to Enabled, then save and exit (usually `F10`).
3. Back in Windows, open PowerShell **as Administrator** and run these two commands. The first is the Virtual Machine Platform feature WSL2 needs. The second is the Windows Hypervisor Platform feature the Android emulator uses.
   ```
   dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   dism /online /enable-feature /featurename:HypervisorPlatform /all /norestart
   ```
4. Reboot once more.
5. Come back to this Claude session and say "continue". You don't need to open Ubuntu yourself; I'll register it.

After that I can carry on through Phases 2 to 8 without stopping, unless TensorFlow can't use your GPU (see the note under Phase 0).

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

## Phase 0b: Environment re-check after your setup (2026-10-23)

- Android: `flutter doctor` is now clean for Android (SDK 36.0.0, licences accepted). The AVD `Medium_Phone_API_37.0` exists.
- WSL: version 3.0.1 and the Ubuntu Store app (2604.1.75.0) are installed, but no distro is registered because the app was never opened. I tried registering it non-interactively with `ubuntu.exe install --root`, which needs no admin rights. It failed with `0x80370102`.
- Root cause: `Win32_Processor.VirtualizationFirmwareEnabled = False`, so SVM is disabled in the BIOS. The CPU itself supports it (`VMMonitorModeExtensions = True`, `SecondLevelAddressTranslationExtensions = True`).
- The emulator is blocked by the same thing: `emulator -accel-check` returns code 6, no hypervisor.
- Next: once SVM and the two Windows features are on, I'll register Ubuntu, check that `nvidia-smi` works inside WSL, and start Phase 2.
- Note for later: if I register Ubuntu as root, your first Ubuntu window will open as root. If you want your own Linux user afterwards, run `adduser <name>` inside Ubuntu, then `ubuntu.exe config --default-user <name>` in PowerShell.
