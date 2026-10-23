# BirdLens progress log

## Final summary (2026-10-23)

**What was built:** BirdLens, a Flutter Android app that identifies 200 bird species from a photo fully on the device. It uses an EfficientNetV2-B0 model I trained on CUB-200-2011 under WSL2 on the RTX 5060 and shipped as a float16 TFLite file. It has camera and gallery input, top-3 results with confidence bars, a "Not confident" message below 0.44, local scan history with delete, a benchmark screen and an About screen. The training pipeline, evaluation, exports, tests, model card and README are all in the repo, committed phase by phase as Lucas Lee, with no remote.

**Final metrics** (official CUB test set, 5,794 images, each .tflite run through the LiteRT interpreter by `training/evaluate.py`):

| Variant | Size | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|---|
| float32 | 24.5 MB | 75.2% | 92.8% | 0.752 |
| float16 (in the app) | 12.4 MB | 75.1% | 92.8% | 0.752 |
| int8 | 7.4 MB | 73.0% | 92.3% | 0.732 |

Best validation accuracy was 77.2%. On the emulator, the phone matches Python to within 1.2e-6 on lossless inputs. Average inference was 51 to 57 ms (Android emulator on my desktop PC, not a real phone).

**Tests:** Python 33 passed, Flutter 45 passed, device integration 11 passed, and `flutter analyze` is clean.

**Where the APK is:** `C:\Users\Admin\birdlens\dist\birdlens-1.0.0-arm64-v8a.apk` (for most phones) and `dist\birdlens-1.0.0-universal.apk`. They're gitignored, ready for GitHub Releases.

**Please check by hand:**
1. **Install the APK on a real phone** and try the camera button. The emulator run only used the gallery, and real-phone speed is untested.
2. **Fixture and sample images are downsized CUB test photos**: 4 in `app/test/fixtures/images/`, 3 in `app/assets/samples/`, plus the figure `docs/misclassifications.png`. CUB is for non-commercial research. A student portfolio fits that, but if you'd rather have none in a public repo, swap in your own bird photos and rerun `training/prepare_app_assets.py`.
3. **Replace the GitHub link placeholder** `https://github.com/your-username/birdlens` in `app/lib/screens/about_screen.dart` once you create the repo.
4. **The release APK is signed with Flutter's debug key.** That's fine for GitHub Releases and sideloading, but create a proper keystore before any Play Store upload.
5. **Building in a new terminal:** AVG's HTTPS scanning breaks Gradle downloads. Run `. $env:USERPROFILE\.birdlens\android-build-env.ps1` first (see Phase 2), or turn off AVG's HTTPS scanning.
6. **Disk space:** C: has about 5.6 GB free. The WSL disk holds the dataset, venv and checkpoints. Per-epoch checkpoints in `~/birdlens-work/checkpoints` can be deleted if you need space; `best.keras` is the one that matters. The WSL virtual disk won't shrink on its own, though.
7. **WSL runs as root.** I registered Ubuntu with `ubuntu.exe install --root` so you didn't have to set up a user. To add your own user, see Phase 0b.
8. The genus-only CUB class names "Sayornis" and "Geococcyx" are shown as-is. You could rename them to "Phoebe" and "Roadrunner" in `training/birdlens_train/labels.py`.

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

## Phase 4: Evaluation and export (2026-10-23)

- `training/export_tflite.py` rebuilds the model in plain float32 (training used mixed precision), loads the best weights and converts three variants: float32 (24.5 MB), float16 (12.4 MB), and full int8 with uint8 input and output (7.4 MB, calibrated on 300 random training images).
- `training/evaluate.py` runs every `.tflite` file through the LiteRT interpreter on all 5,794 test images and 600 validation images (`training/results/tflite_metrics.json`):
  - float32: top-1 **75.18%**, top-5 92.84%, macro F1 0.752. Its val top-1 is 77.17%, identical to the Keras checkpoint, which confirms the conversion and the eval pipeline.
  - float16: top-1 **75.15%**, top-5 92.84%, macro F1 0.752. Same top-1 answer as float32 on 99.91% of test images.
  - int8: top-1 **73.04%**, top-5 92.34%, macro F1 0.732. Agrees with float32 on 88.21%. It was about 1.7x faster in Python on the CPU (32.4 s against 54.3 s for the test set).
- Chose **float16** for the app. It keeps float32 accuracy at half the size. Int8 saves 5 MB more but costs 2.1 points of top-1.
- Threshold (`training/choose_threshold.py`): **0.44**, the lowest threshold with at least 90% accuracy above it on validation. Validation: 68.2% coverage at 90.2% accuracy. Test, same threshold: 69.0% coverage at 88.1% accuracy, and 46.2% accuracy below it. Chart: `docs/threshold_curve.png`.
- Error analysis (`training/analyze_errors.py`): the worst classes are Western Wood Pewee (13%), Brewer's Blackbird (14%) and Common Tern (17%). The most confused pairs are jaegers, vireos, cormorants and terns. The grid is `docs/misclassifications.png`, and the full lists are in `training/results/error_analysis.json` and `docs/model_card.md`.
- `training/prepare_app_assets.py` copies the float16 model, `labels.txt` and `model_config.json` into `app/assets/`, makes 3 benchmark sample photos, and writes the parity fixtures. The fixtures are 4 test images as PNG and JPEG, downsized to 320 px, with the Python preprocessing values and the outputs of every variant. They go in `app/test/fixtures/` and are embedded for the device test in `app/integration_test/fixture_data.dart`.
- Label cleaning: I checked all 200 cleaned names against `classes.txt` by eye. I fixed the possessives CUB drops (Anna's Hummingbird, Forster's Tern and others), "Whip-poor-will", and CUB's own typo "Artic Tern", which became "Arctic Tern". The genus-only CUB classes "Sayornis" and "Geococcyx" are left as they are.
- `training/scripts/count_params.py`: 6,175,512 parameters in total (5,919,312 backbone, 256,200 head).
- `docs/model_card.md` is written: intended use, the dataset licence (Flickr, non-commercial research, not redistributed), metrics, limitations and fairness of the evaluation.

## Phase 6: Testing (2026-10-23)

- Python (`python -m pytest` in `training/`): **33 passed**. They cover label cleaning (including every special case), the official split loading, stratified and deterministic validation split, the real split file sizes (5,994 and 5,794, no overlap, all 200 classes in val), preprocessing (shape, range, greyscale, and a match against my own reference bilinear within 0.02), and the metrics (top-k, confusion matrix, macro F1, confused pairs, threshold table).
- Flutter (`flutter test`): **45 passed**, with `flutter analyze` reporting no issues. They cover:
  - Unit tests for preprocessing (bilinear against hand-worked TensorFlow values, decode, greyscale, alpha, bad bytes, quantisation), result parsing, dequantisation, label parsing, the ScanRecord database round trip, benchmark stats and formatting.
  - Widget tests for the identify screen (loaded, model failed, camera denied, picker cancelled), the result screen (confident, not confident, corrupt image), history (empty, order, delete), benchmark, about and navigation.
  - Host parity test against the Python fixtures:
    - PNG: Dart matches Python within 0.02 per value.
    - JPEG: a mean difference under 2 is allowed, because Dart's JPEG decoder differs from libjpeg-turbo. I measured 1.2 to 1.4 mean and up to 16 at single edge pixels, and the tolerances and the reasons are written in the test.
- On the emulator (`flutter test integration_test -d emulator-5554`): **11 passed**.
  - 9 parity tests. PNG fixtures: largest probability difference from Python 5.3e-7 to 1.2e-6. JPEG fixtures: same top-1 on all 4, with a top-1 probability difference of 0.003 to 0.027.
  - 2 stress tests: 200 inferences in a row, and loading and closing the model twice.

## Phase 7: Emulator, screenshots and release APK (2026-10-23)

- Installed the missing pieces with `sdkmanager` through the AVG-aware trust store: NDK 28.2, and the Android 37 x86_64 Play system image. The image folder had been left empty by the Android Studio setup.
- Disk: C: had only about 10.7 GB free and the AVD wanted a 12 GB data partition. I launched it with `-partition-size 2047`, a launch flag only, so your AVD config is unchanged. Free space later fell to 2.8 GB. `flutter clean` freed 2 GB of my own build output, leaving about 5.6 GB.
- Pushed 5 CUB test photos to `/sdcard/Pictures/BirdLens` and ran a media scan. They were chosen by `training/scripts/pick_demo_images.py`: 4 confidently correct ones plus 1 real low-confidence case, a Black-footed Albatross guessed as Sooty Albatross at 19%. I drove the app with `adb shell input` and checked every screenshot myself.
- UI fixes from looking at the screenshots:
  - Removed the grey bars around the photo on the result screen.
  - Shortened history timestamps so lines don't wrap.
  - Sizes now use decimal MB to match the README.
  - Made the benchmark caption accurate.
  - Updated the About screen's look-alike examples to match the real error analysis.
- **A real bug found on the device:** one benchmark run failed with "Unsupported output type noType". tflite_flutter's `IsolateInterpreter` shares one native interpreter between two isolates and re-allocates its tensors every call, while my code read tensor details on the main isolate. Moving the interpreter into my own worker isolate (`app/lib/services/inference_worker.dart`) uncovered the root cause: tensor handles fetched before the first `invoke()` go stale on this runtime. I saw a 0-byte input and a SIGSEGV in `TfLiteTensorCopyToBuffer` on a null buffer. A small on-device probe confirmed that fresh handles each call work in every setup. The worker now fetches tensors fresh every call, and a 200-run stress test guards it.
- Checked by hand on the emulator: gallery pick, top-3 bars, the not-confident banner, saving to history, history surviving force-stop and reinstall, delete with confirmation, the benchmark, and the About screen. The camera button wasn't exercised, because the emulator's camera needs a GUI. The denied-camera dialog is covered by a widget test only.
- Benchmark, labelled **Android emulator on my desktop PC, not a real phone**, 3 cold starts (`docs/emulator_results.md`): warm-up 511.4, 441.2 and 433.1 ms. Average 50.6, 57.3 and 53.6 ms. p95 61.3, 88.3 and 66.3 ms.
- Screenshots: `docs/screenshots/1_home.png` to `6_about.png`, downsized to 540x1200.
- Release APKs: `dist/birdlens-1.0.0-arm64-v8a.apk` (37.2 MB) and `dist/birdlens-1.0.0-universal.apk` (84.1 MB). Both are gitignored, for you to upload to GitHub Releases. SHA-256 checksums:
  - arm64-v8a: `83454334991bb57316bffa89e8e179fc94052c92e2bbf814469330c0dc639f48`
  - universal: `c98aed0cfae746469c6992f2b0b4f3f3bfa2fd3604eb4794e4289b139419ddf2`
- The emulator was shut down afterwards to give your PC its memory back. While it was running, WSL briefly refused to start ("insufficient system resources").

## Phase 8: README (2026-10-23)

- `README.md`: the one-line summary, a screenshot row, the problem and what the app does, the results table, how it works with a Mermaid diagram (data, model choice, training, quantisation, on-device inference, parity, threshold), evaluation with the confusion highlights and misclassification grid, limitations, what I'd do next, setup for training and the app, Releases, and the author. Every number names the script or file it came from.
- Checked the README, docs and code comments: no em dashes, none of the listed AI-sounding words, and no all-caps emphasis in comments (only acronyms like JPEG and EXIF).
