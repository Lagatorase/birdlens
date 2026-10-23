# BirdLens progress log

## Status

Work in progress. No action needed from you right now. The ACTION NEEDED about SVM was resolved on 2026-10-23.

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
- Ubuntu ships a newer Python than 3.11, so I get Python 3.11 through `uv` without needing sudo.

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

## Phase 0c: WSL up and GPU check (2026-10-23)

- After you enabled SVM: `VirtualizationFirmwareEnabled = True`. I registered the Ubuntu app with `ubuntu.exe install --root`, which needed no admin rights. It came up as **Ubuntu 26.04.1 LTS** under WSL2.
- `nvidia-smi` inside WSL: **NVIDIA GeForce RTX 5060, 8151 MiB**, driver 591.44, CUDA 13.1 driver API. WSL has 12 CPU threads and about 7 GB RAM.
- The emulator now reports `WHPX(10.0.19045) is installed and usable`.
- Midway through, WSL auto-updated itself. For a few minutes `wsl` failed with `ERROR_SERVICE_DISABLED` and `ERROR_INSTALL_ALREADY_RUNNING`, which killed my dataset download. It recovered without any action from me, and the venv and partial download survived.

## Phase 2: Training environment and data (2026-10-23)

- Python 3.11.17 venv built with `uv` at `~/birdlens-work/.venv` and symlinked to `training/.venv` (`training/scripts/setup_env.sh`).
- TensorFlow: I checked PyPI rather than guessing. **TF 2.21.0** is the newest stable release (2026-03-06, with 2.22 still at rc0) and has cp311 wheels. `tensorflow[and-cuda]==2.21.0` pulls the CUDA 12.9 and cuDNN 9.27 pip wheels. I tested `tf.lite.TFLiteConverter.from_keras_model` on a Keras 3.15.1 model before committing to this version, and it works. `tf.lite.Interpreter` prints a deprecation warning, so evaluation uses `ai-edge-litert==2.3.0` instead. Exact pins are in `training/requirements.txt`, and the full resolved set is in `training/requirements.lock.txt`.
- GPU problem 1, fixed: TF couldn't `dlopen libcusolver.so.11` even though the pip wheel ships it. Fix: `training/scripts/env.sh` puts every `nvidia/*/lib` folder from the venv on `LD_LIBRARY_PATH`. Every training command sources it.
- GPU problem 2, worked around: TF 2.21 has no prebuilt kernels for compute capability 12.0 (Blackwell). It warns and JIT-compiles from PTX. I raised `CUDA_CACHE_MAXSIZE` so the JIT cache persists between runs. Smoke test (`training/scripts/gpu_check.py`): EfficientNetV2-B0, batch 16, mixed precision, 10 train steps in 0.27 s, peak 402 MiB with the 4000 MiB cap active.
- Dataset: the official CaltechDATA record 65de6-vp158, `CUB_200_2011.tgz`, MD5 `97eceeb196236b17998738112f37df78` (`training/scripts/download_cub.sh`). The server is slow and sometimes stalls at a few KB/s, so the script now aborts and resumes whenever the speed stays under 50 KB/s for a minute. Note: the CaltechDATA page lists the record licence as "cc-by", but the images themselves come from Flickr, and the dataset's own terms are non-commercial research use. The model card says this.
- Android build blocker, worked around: the first `flutter build apk` failed downloading Gradle with `PKIX path building failed`. The cause is **AVG Antivirus HTTPS scanning**. On Windows, the certificate for `services.gradle.org` is issued by "AVG Web/Mail Shield Root", which Windows trusts but Android Studio's bundled Java (JBR 25) does not. WSL isn't affected. The JBR also lacks the `Windows-ROOT` store provider, so pointing Java at the Windows certificate store failed too. Fix: I exported the AVG root certificate from Windows (a read-only step, no admin) and built `C:\Users\Admin\.birdlens\truststore.p12`, which is a copy of JBR's `cacerts` plus that one certificate. My build commands point Java at it through `JAVA_TOOL_OPTIONS`. Nothing in AVG, Java or Windows was changed. **When you build in a new PowerShell window, first run** `. $env:USERPROFILE\.birdlens\android-build-env.ps1`, or turn off HTTPS scanning in AVG yourself if you prefer.

## Phase 5 (part): Flutter app built (2026-10-23)

- App structure: `lib/models`, `lib/services` (preprocessing, classifier, result parsing, history, image storage, photo source, benchmark), `lib/screens` (identify, result, history, benchmark, about, navigation shell), `lib/widgets`, `lib/utils`. Material 3 with a forest-green seed colour, plus light and dark themes.
- Package versions checked on pub.dev: tflite_flutter 0.12.1 (uses LiteRT 1.4.0 on Android, needs API 26 or newer), image_picker 1.2.4, sqflite 2.4.5, image 4.12.0, path_provider 2.1.6, url_launcher 6.3.3. I set `minSdk = 26`.
- Things I found in the tflite_flutter source and worked around in `lib/services/classifier.dart`:
  - `IsolateInterpreter.run` silently returns without output if another run is in flight, so my classifier queues runs one at a time.
  - Passing a `Uint8List` input skips the slow nested-list conversion, so I pass raw tensor bytes and read the output through a `ByteBuffer`.
- Android build fixes, beyond the AVG trust store above:
  1. Gradle couldn't auto-install NDK 28.2.13676358, so I installed it with `sdkmanager`.
  2. `permission_handler_android` 14.1.0 needs `compileSdk = 37`, which this AGP can't resolve against the SDK's `android-37.0` folder. I removed permission_handler. `image_picker` already requests the CAMERA permission declared in the manifest and reports a denial as `camera_access_denied`, which the app turns into a "Camera access is off" dialog.
  3. tflite_flutter's Java targets JVM 11 but its Kotlin defaulted to 25. I pinned its Kotlin `jvmTarget` to 11 in `app/android/build.gradle.kts`.
- `flutter analyze`: no issues. `flutter test`: 35 passed (unit and widget). The parity test is skipped until the fixtures exist. The debug APK builds.

## Phase 3: Training (2026-10-23)

- Model: EfficientNetV2-B0 (ImageNet weights, built-in 0 to 255 input rescaling), global average pooling, dropout 0.3, Dense(200) softmax in float32. Input 224x224, mixed precision, batch 16, label smoothing 0.1.
- Light GPU mode: memory growth on plus a 4000 MiB logical-device cap (about half of 8151 MiB), `nice -n 10`, 4 tf.data workers with a 4-thread private pool. The PTX JIT compile makes the first epoch of each stage slow: 153 s and 140 s. After that, epochs took 7 to 18 s. GPU use was about 30% and the training process used about 2.5 of 12 CPU threads.
- Stage 1 (head only, Adam 1e-3, early stopping patience 4): 8 epochs, best val accuracy **68.17%** at epoch 4.
- Stage 2 (top 40% of backbone layers unfrozen, 87 layers, batch norm kept frozen; AdamW, cosine decay from 1e-4, weight decay 1e-5; patience 6): 21 epochs, best val accuracy **77.17%** (top-5 **94.33%**) at epoch 15. Training accuracy reached 99%, so the model clearly overfits the 5,394 training images. Validation plateaued around 75 to 77% from epoch 6 onwards.
- Source: `training/results/train_summary.json`, written by `training/train.py`. Per-epoch numbers are in `training/logs/*_history.csv`. Total time was about 10 minutes of epochs (217 s plus 366 s).
- A bug I caught: the first summary reported 76.25% val accuracy, not the 77.17% Keras logged for the best epoch. Re-scoring the saved `best.keras` and `epoch_15.keras` in a fresh process (`training/scripts/check_checkpoints.py`) gave 77.17% under both float32 and mixed precision. So the weights were fine, and the in-process re-evaluation after training was the problem. `train.py` now clears the Keras session first and gives the same 77.17%.
- Resumability checked with `training/scripts/resume_test.sh`, which runs in a throwaway folder: I killed a run with `kill -9` during head epoch 3, and the restart continued at "Epoch 3/4" with no duplicate history rows. A rerun after both stages are done skips straight to the summary.
- Next: export the TFLite variants and evaluate them.
