# BirdLens

[![Tests](https://github.com/Lagatorase/birdlens/actions/workflows/tests.yml/badge.svg)](https://github.com/Lagatorase/birdlens/actions/workflows/tests.yml)
[![Latest release](https://img.shields.io/github/v/release/Lagatorase/birdlens)](https://github.com/Lagatorase/birdlens/releases/latest)
[![License: MIT](https://img.shields.io/github/license/Lagatorase/birdlens)](LICENSE)
![Flutter 3.47](https://img.shields.io/badge/Flutter-3.47-02569B?logo=flutter&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-on--device-FF6F00?logo=tensorflow&logoColor=white)
![Android 8.0+](https://img.shields.io/badge/Android-8.0%2B-3DDC84?logo=android&logoColor=white)

An Android app that identifies 200 bird species from a photo, entirely on the phone, using a TensorFlow Lite model I trained on CUB-200-2011.

<p>
  <img src="docs/screenshots/1_home.png" width="160" alt="Home screen">
  <img src="docs/screenshots/2_result.png" width="160" alt="Result for a Painted Bunting">
  <img src="docs/screenshots/3_not_confident.png" width="160" alt="Low confidence result">
  <img src="docs/screenshots/4_history.png" width="160" alt="Scan history">
  <img src="docs/screenshots/5_benchmark.png" width="160" alt="Benchmark screen">
</p>

**[Download the APK (v1.0.0)](https://github.com/Lagatorase/birdlens/releases/tag/v1.0.0)** for Android 8.0 or newer.

## The problem

Most bird ID apps upload your photo to a server. I wanted to see how well a small model can do running only on the phone: no upload, works offline, quick enough to feel instant. That meant training my own classifier, shrinking it for mobile, and proving the phone gives the same answers as my Python evaluation.

## What the app does

- Take a photo or choose one from the gallery, and see the top 3 species with confidence bars.
- If the top guess is weak, it says "Not confident, this may not be one of the 200 species I know" instead of pretending.
- Keeps a local history of scans (thumbnail, top guess, confidence, time) that you can delete.
- A benchmark screen times the model on the device. An About screen explains what the model can't do.

## Results

Measured on the official CUB-200-2011 test set (5,794 photos), running each `.tflite` file through the LiteRT interpreter ([training/evaluate.py](training/evaluate.py), output in [training/results/tflite_metrics.json](training/results/tflite_metrics.json)):

| Model file | Size | Top-1 accuracy | Top-5 accuracy | Macro F1 |
|---|---|---|---|---|
| float32 | 24.5 MB | 75.2% | 92.8% | 0.752 |
| **float16 (in the app)** | **12.4 MB** | **75.1%** | **92.8%** | **0.752** |
| int8 | 7.4 MB | 73.0% | 92.3% | 0.732 |

On the Android emulator on my desktop PC (not a real phone), one inference with the float16 model took 50.6 to 57.3 ms on average across three runs of the app's benchmark, after a cold warm-up of 433 to 511 ms ([docs/emulator_results.md](docs/emulator_results.md)). Real phones will differ.

## How it works

```mermaid
flowchart LR
    subgraph Training ["Training (WSL2, RTX 5060)"]
        A[CUB-200-2011<br/>official split] --> B[Augment<br/>crop, flip, colour]
        B --> C[EfficientNetV2-B0<br/>ImageNet weights]
        C --> D[Stage 1: train head<br/>backbone frozen]
        D --> E[Stage 2: fine-tune<br/>top 40% of backbone]
    end
    E --> F[TFLite converter<br/>float32 / float16 / int8]
    F --> G[LiteRT evaluation<br/>on the test set]
    F --> H[Flutter app]
    subgraph Phone ["On the phone"]
        H --> I[Photo, max 512 px] --> J[Decode, rotate,<br/>bilinear resize to 224]
        J --> K[TFLite interpreter<br/>in a worker isolate] --> L[Top 3 + threshold]
    end
```

**Data.** CUB-200-2011 has 11,788 photos of 200 mostly North American species. I used the official train/test split and held out 10 percent of each class from the training set for validation, with a fixed seed ([training/make_splits.py](training/make_splits.py)). The test set wasn't used for any decision. Early stopping, checkpoint choice and the confidence threshold all used validation.

**Model choice.** I picked **EfficientNetV2-B0** over MobileNetV3-Large. Telling 200 similar-looking birds apart is a fine-grained task, and EfficientNetV2-B0 is the more accurate ImageNet backbone of the two while still being small: my full model has 6.2M parameters ([training/scripts/count_params.py](training/scripts/count_params.py)). It also has its 0 to 255 rescaling built into the Keras model. So the app feeds raw RGB pixels, and there is no normalisation step for Dart and Python to disagree on. The cost is speed: it does more work per image than MobileNetV3, which the benchmark numbers reflect.

**Training.** Two stages ([training/train.py](training/train.py)). First I trained only the new 200-way head with the backbone frozen, so random head weights couldn't wreck the pretrained features. Then I unfroze the top 40 percent of the backbone (batch norm stayed frozen) and fine-tuned with a 10x lower learning rate and cosine decay. I used mixed precision, batch 16 and label smoothing 0.1. Augmentation was random crops, horizontal flips and mild colour jitter, with no vertical flips because birds aren't upside down. Training ran in "light mode" next to a game: VRAM capped at about half the card, `nice -n 10`, and 4 data workers. Each epoch saves a checkpoint and the script resumes from the last one if interrupted, which I tested by killing a run mid-epoch. Best validation accuracy was 77.2% after about 10 minutes of training ([training/results/train_summary.json](training/results/train_summary.json)).

**Quantisation.** I exported three TFLite files ([training/export_tflite.py](training/export_tflite.py)) and the app ships **float16**. It's half the size of float32, its accuracy is the same to within 0.03 points, and it gives the same top-1 answer as float32 on 99.9% of test photos. Full int8, calibrated on 300 training photos, would save another 5 MB and ran about 1.7x faster in my Python CPU evaluation. But it loses 2.1 points of top-1 and changes the answer on 11.8% of photos compared with float32. For a 12 MB model the size saving didn't seem worth that.

**On-device inference.** The app uses `tflite_flutter`. Preprocessing in Dart ([app/lib/services/preprocessing.dart](app/lib/services/preprocessing.dart)) copies the Python eval path exactly: decode to RGB, apply the EXIF rotation, resize the whole photo to 224x224 with half-pixel bilinear sampling, and keep pixels at 0 to 255. Photos come from the picker at most 512 px on the long side, close to the size of CUB images, so the plain bilinear resize sees birds at a similar scale to training. Inference runs in a long-lived worker isolate ([app/lib/services/inference_worker.dart](app/lib/services/inference_worker.dart)) so the UI never freezes.

**Proving the phone matches Python.** [training/prepare_app_assets.py](training/prepare_app_assets.py) saves four test images with the Python outputs for each. On the emulator, [app/integration_test/parity_test.dart](app/integration_test/parity_test.dart) runs the Dart preprocessing and the real TFLite runtime on them:

- With lossless PNG inputs, the largest probability difference from Python is 1.2e-6.
- With JPEG inputs (different decoders on each side), the top-1 species matches on all four, and its probability differs by 0.003 to 0.027.

**Confidence threshold.** The "Not confident" message appears when the top-1 probability is under **0.44**. I chose it on the validation set as the lowest threshold where at least 90% of the answers above it are correct ([training/choose_threshold.py](training/choose_threshold.py)). On the test set, 69% of photos clear it and 88.1% of those are right, while photos below it are right only 46% of the time. See the [threshold curve](docs/threshold_curve.png). This threshold only reflects in-distribution photos. A bird that isn't one of the 200 species can still get a confident wrong answer.

## Evaluation

The model is weakest on species that are hard for people too ([training/analyze_errors.py](training/analyze_errors.py)). The most confused pairs on the test set:

| Pair | Mix-ups (both directions) |
|---|---|
| Long-tailed Jaeger and Pomarine Jaeger | 15 |
| Philadelphia Vireo and Warbling Vireo | 15 |
| Brandt's Cormorant and Pelagic Cormorant | 14 |
| Arctic Tern and Common Tern | 14 |
| Common Tern and Forster's Tern | 13 |

The worst single classes are Western Wood Pewee (13% correct, usually called an Olive-sided Flycatcher), Brewer's Blackbird (14%, called a Shiny Cowbird) and Common Tern (17%, called an Arctic Tern). The full lists are in the [model card](docs/model_card.md).

![Confident mistakes from the most confused pairs](docs/misclassifications.png)

## Out-of-scope photos

The threshold was tuned on CUB photos only, so I checked what it does on photos the model was never meant to see. I built two sets of 300 photos each ([training/scripts/build_out_of_scope.py](training/scripts/build_out_of_scope.py)). The first has birds of 20 species that aren't among the 200: ten North American (American Robin, Mourning Dove, Bald Eagle and so on) and ten from elsewhere (European Robin, Rainbow Lorikeet, Laughing Kookaburra and so on), 15 photos each, from research-grade iNaturalist observations licensed CC0 or CC BY. The second has photos with no bird: animals, plants, objects and scenes from the Open Images V7 validation set (CC BY 2.0), skipping any photo with a bird label. The manifests in [training/out_of_scope/](training/out_of_scope/) list every photo's source, licence and author. I ran the app's float16 file through the LiteRT interpreter with the same preprocessing as `evaluate.py` ([training/evaluate_out_of_scope.py](training/evaluate_out_of_scope.py), output in [training/results/out_of_scope.json](training/results/out_of_scope.json)). The threshold stayed at 0.44.

| Photo set | Photos | "Not confident" shown | Median top-1 probability |
|---|---|---|---|
| CUB test set (for comparison) | 5,794 | 31.0% | 0.64 |
| Birds of species not in CUB | 300 | 88.7% | 0.17 |
| Photos with no bird | 300 | 98.3% | 0.07 |

![Top-1 confidence on the CUB test set and the two out-of-scope sets](docs/out_of_scope_confidence.png)

The message catches almost every photo with no bird: 5 of 300 got a confident answer, all between 0.46 and 0.55, and four of them were gulls (two road photos, a car and a sea view). Birds outside the 200 species are harder. 34 of 300 (11.3%) still got a confident wrong species, and one of those was above 0.9. The worst were House Finch (7 of 15 confident) and Rainbow Lorikeet (8 of 15), and the commonest confident answers were Painted Bunting (8) and Purple Finch (4). A caveat: many iNaturalist photos show small, distant birds, which also lowers confidence on in-scope species, so some of the 88.7% comes from harder photos rather than unknown species.

## Limitations

- It only knows the 200 CUB species. Any other bird gets labelled as one of those 200, sometimes confidently (11.3% of my out-of-scope bird photos got a confident wrong answer).
- CUB photos are mostly clear, close shots. Blurry, distant, backlit or partly hidden birds will do worse than the test numbers suggest.
- With about 27 training photos per species, the model overfits (99% training accuracy against 77% validation).
- The benchmark numbers come from an emulator, not a real phone.
- The release APKs are signed with Flutter's default debug key, which is fine for sideloading but not for a store release.

## What I'd do next

- Test on real phones and real field photos, not just CUB test images.
- Add an "is this a bird at all" check, or train with an extra "other" class, so out-of-scope photos are caught more reliably than by a threshold.
- Try a bird detector to crop the bird before classifying. That should help most with small, distant birds.
- Train on a larger dataset such as NABirds, which has more photos per species and covers more species.
- Compare MobileNetV3 and int8 speed on actual ARM phones, where int8 kernels tend to matter more.

## Project structure

```text
birdlens/
├── app/                      Flutter app
│   ├── assets/               float16 model, labels, config, benchmark photos
│   ├── lib/
│   │   ├── models/           data classes (predictions, scans, model info)
│   │   ├── screens/          identify, result, history, benchmark, about
│   │   ├── services/         preprocessing, TFLite worker, history, photo picking
│   │   ├── utils/            formatting helpers
│   │   └── widgets/          confidence bars, banners, empty and error states
│   ├── test/                 unit, widget and preprocessing parity tests
│   └── integration_test/     on-device parity and stress tests
├── training/                 Python training pipeline
│   ├── birdlens_train/       shared code: data, model, preprocessing, metrics
│   ├── scripts/              environment setup, dataset download, checks
│   ├── splits/               the exact train, val and test split used
│   ├── out_of_scope/         manifests of the out-of-scope photo sets
│   ├── results/              metrics JSON quoted in this README
│   └── tests/                pytest tests
└── docs/                     model card, emulator results, screenshots, figures
```

## Getting started

### Install the app

Download `birdlens-1.0.0-arm64-v8a.apk` from the [v1.0.0 release](https://github.com/Lagatorase/birdlens/releases/tag/v1.0.0) and install it on an Android 8.0+ phone. Use the universal APK on the same page if you're not sure which one your phone needs.

### Prerequisites

To build the app:

- Flutter 3.47 or newer (tested with 3.47.7)
- The Android SDK, through Android Studio

To reproduce the training:

- WSL2 or Linux with an NVIDIA GPU (I used an RTX 5060 with 8 GB, capped at about 4 GB)
- About 10 GB of free disk for the dataset, Python environment and checkpoints
- `curl`, plus internet access to download CUB-200-2011 from CaltechDATA

### Reproduce the training

```bash
cd training
bash scripts/setup_env.sh        # Python 3.11 venv with the pinned packages
bash scripts/download_cub.sh     # CUB-200-2011 from CaltechDATA, MD5 checked
source scripts/env.sh            # CUDA library paths for TensorFlow
python make_splits.py            # official split + 10% stratified validation
nice -n 10 python train.py       # two-stage training, resumable
python export_tflite.py          # float32, float16 and int8 .tflite files
python evaluate.py               # LiteRT evaluation on test and validation
python choose_threshold.py --variant float16
python analyze_errors.py --variant float16
python prepare_app_assets.py --variant float16   # copies the model into the app
python scripts/build_out_of_scope.py             # downloads the out-of-scope photo sets
python evaluate_out_of_scope.py                  # "Not confident" rate on those sets
python -m pytest                 # label cleaning, split, preprocessing, metrics
```

Versions are pinned in [training/requirements.txt](training/requirements.txt) (TensorFlow 2.21.0, Keras 3.15.1, ai-edge-litert 2.3.0). The full resolved set is in [training/requirements.lock.txt](training/requirements.lock.txt).

### Run the app

```bash
cd app
flutter pub get
flutter test                      # unit, widget and preprocessing parity tests
flutter test integration_test     # on a running emulator or phone
flutter run
```

The model, labels and config are already committed in `app/assets/`, so you don't need to train anything to run the app.

## Acknowledgements

- **CUB-200-2011:** C. Wah, S. Branson, P. Welinder, P. Perona and S. Belongie. *The Caltech-UCSD Birds-200-2011 Dataset.* California Institute of Technology, Technical Report CNS-TR-2011-001, 2011.
- **EfficientNetV2:** M. Tan and Q. V. Le. *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021. I used the ImageNet weights shipped with Keras.
- **Out-of-scope photos:** bird photos by iNaturalist observers (CC0 or CC BY) and photos from the [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) (CC BY 2.0). Each photo's author and licence are in [training/out_of_scope/](training/out_of_scope/).
- [tflite_flutter](https://pub.dev/packages/tflite_flutter) for running TensorFlow Lite from Dart, and [LiteRT](https://ai.google.dev/edge/litert) for the Python interpreter used in evaluation.

## License

My code is released under the [MIT License](LICENSE).

The CUB-200-2011 images are not part of this repository, apart from a few downsized test images used for fixtures and figures. They come from Flickr and are for non-commercial research and educational use only. See the [model card](docs/model_card.md) for details. The out-of-scope photos aren't in the repository either, only manifests with their sources, licences and authors.

## Author

Lucas Lee, Temasek Polytechnic.
