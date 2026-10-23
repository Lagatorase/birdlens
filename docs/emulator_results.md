# Emulator results

All numbers on this page come from the **Android emulator on my desktop PC, not a real phone**. Expect different numbers on actual phones. The emulator runs an x86_64 Android image directly on my desktop CPU, while most phones have ARM CPUs, and it shares that CPU with everything else running on the PC.

## Setup

- Host: AMD Ryzen 5 5600G, 16 GB RAM, Windows 10, emulator 37.2.12 with WHPX acceleration
- AVD: Medium_Phone_API_37.0 (Android 17, x86_64, 1080x2400), started with `-memory 3072 -cores 4 -partition-size 2047`
- App: release build of BirdLens 1.0.0, float16 model (12.4 MB), TFLite interpreter with 4 threads and the default CPU (XNNPACK) path

## Benchmark screen

The app's Benchmark screen (`app/lib/services/benchmark.dart`) preprocesses three bundled sample photos once. It then times one warm-up inference and 50 more, cycling through the samples. Times are the interpreter's `invoke()` call inside the worker isolate, so photo decoding and resizing are not included. I force-stopped the app before each run so the warm-up is a cold first inference.

| Run | Warm-up (first run) | Average | p95 | Fastest |
|---|---|---|---|---|
| 1 | 511.4 ms | 50.6 ms | 61.3 ms | 39.8 ms |
| 2 | 441.2 ms | 57.3 ms | 88.3 ms | 40.3 ms |
| 3 | 433.1 ms | 53.6 ms | 66.3 ms | 43.0 ms |

The values were read from the screen with `adb shell uiautomator dump`. Screenshot of run 1: `docs/screenshots/5_benchmark.png`.

## On-device parity test

`app/integration_test/parity_test.dart` runs on the emulator. It feeds the four fixture images through the app's Dart preprocessing and the real tflite_flutter runtime, then compares the output with the Python LiteRT interpreter's output for the same files (`app/test/fixtures/fixtures.json`, written by `training/prepare_app_assets.py`).

| Fixture | PNG: largest probability difference | JPEG: same top-1? | JPEG: top-1 probability difference |
|---|---|---|---|
| fixture_1 | 5.27e-7 | yes | 0.0029 |
| fixture_2 | 5.57e-7 | yes | 0.0148 |
| fixture_3 | 9.60e-7 | yes | 0.0266 |
| fixture_4 | 1.20e-6 | yes | 0.0094 |

The PNG versions give both sides identical pixels, so the tiny gap is float rounding between the two runtimes. The JPEG versions also include the difference between Dart's JPEG decoder and libjpeg-turbo, which `app/test/parity_test.dart` measures at a mean of 1.2 to 1.4 (on the 0 to 255 scale) per pixel.

`app/integration_test/stress_test.dart` also runs 200 inferences in a row and checks that every output is a valid probability vector with the same top-1. It also loads and closes the model twice. All 11 device tests passed.

## APK sizes

From `flutter build apk --release` and `--split-per-abi` (decimal MB, file sizes on disk):

| APK | Size |
|---|---|
| arm64-v8a (most current phones) | 37.2 MB |
| armeabi-v7a (older 32-bit phones) | 32.3 MB |
| x86_64 (emulators) | 40.6 MB |
| Universal (all of the above) | 84.1 MB |

Most of the size is the native TFLite runtime that tflite_flutter bundles (LiteRT plus its GPU delegate) and the Flutter engine. The model itself is 12.4 MB.
