# Model card: BirdLens bird species classifier

## Summary

An image classifier that names the bird species in a photo, chosen from the 200 species in CUB-200-2011. It is an EfficientNetV2-B0 network pretrained on ImageNet and fine-tuned on CUB, then converted to TensorFlow Lite so it runs offline on Android phones in the BirdLens app.

- Author: Lucas Lee, Temasek Polytechnic
- Version: 1.0 (October 2026)
- Code: `training/` in this repository
- Licence of the code and trained weights: MIT. See "Dataset and licence" for the limits that come from the training data.

## Intended use

- A portfolio and learning project showing an end-to-end on-device ML pipeline.
- Casual identification of birds by hobbyists, as a hint rather than an authority.

Not intended for:

- Scientific surveys, conservation decisions, or anything where a wrong species matters.
- Commercial use. The training images are licensed for non-commercial research only.
- Birds outside the 200 CUB species. The model always answers with one of its 200 classes.

## Dataset and licence

- **CUB-200-2011** (Wah, Branson, Welinder, Perona and Belongie, Caltech, 2011): 11,788 photos of 200 bird species, mostly found in North America.
- Downloaded from the official CaltechDATA record (`https://data.caltech.edu/records/65de6-vp158`, MD5 `97eceeb196236b17998738112f37df78`).
- The photos were collected from Flickr. The dataset authors state that the images are for **non-commercial research and educational use**. The CaltechDATA page also lists a "cc-by" licence for the record, but that doesn't change the rights of the original Flickr photographers, so I treat the images as non-commercial research use only.
- **This repository does not redistribute the dataset.** The only CUB-derived images included are a handful of downsized test images: the parity test fixtures in `app/test/fixtures/`, the benchmark samples in `app/assets/samples/`, and the figure `docs/misclassifications.png`. They are there for testing and analysis.
- Split: the official `train_test_split.txt` (5,994 train and 5,794 test images). I hold out 10 percent of each class from the training set as validation, with seed 42 (`training/make_splits.py`). The split files are committed in `training/splits/`.

## Training

Script: `training/train.py`. Results: `training/results/train_summary.json`, with per-epoch logs in `training/logs/head_history.csv` and `training/logs/finetune_history.csv`.

- Backbone: Keras `EfficientNetV2B0`, ImageNet weights, with its built-in rescaling so the model takes raw 0 to 255 RGB.
- Head: global average pooling, dropout 0.3, a 200-way softmax.
- Input: the whole photo resized to 224x224 with bilinear sampling, no crop.
- Augmentation: a random crop covering 50 to 100 percent of the photo (aspect 3:4 to 4:3), horizontal flip, and mild brightness, contrast, saturation and hue jitter. No vertical flips.
- Stage 1: backbone frozen, Adam at 1e-3, label smoothing 0.1, batch 16, mixed precision. It stopped early after 8 epochs, with the best validation accuracy of 68.2% at epoch 4.
- Stage 2: top 40 percent of backbone layers unfrozen (87 layers, batch norm kept frozen), AdamW with cosine decay from 1e-4. It stopped early after 21 epochs, with the best validation accuracy of **77.2%** (top-5 94.3%) at epoch 15.
- Hardware: RTX 5060 under WSL2, capped at 4000 MiB of VRAM. About 10 minutes of training time in total.
- The model overfits: training accuracy reached 99% while validation settled around 75 to 77%. With only about 27 training photos per class, that's expected.

## Evaluation

All on the official CUB test split (5,794 images), with each `.tflite` file run through the LiteRT interpreter (`training/evaluate.py`, results in `training/results/tflite_metrics.json`).

| Variant | File size | Top-1 | Top-5 | Macro F1 | Agrees with float32 |
|---|---|---|---|---|---|
| float32 | 24.5 MB | 75.2% | 92.8% | 0.752 | (reference) |
| float16 (shipped) | 12.4 MB | 75.1% | 92.8% | 0.752 | 99.9% |
| int8 (full integer, uint8 in and out) | 7.4 MB | 73.0% | 92.3% | 0.732 | 88.2% |

The float32 file scores 77.17% on validation, the same as the Keras checkpoint, which suggests the conversion is faithful. Int8 calibration used 300 random training images, preprocessed the same way as inference.

**Confidence threshold** (`training/choose_threshold.py`, `training/results/threshold.json`): the app shows "Not confident" when the top-1 probability is below **0.44**. That's the lowest threshold where predictions at or above it are at least 90% correct on the validation set. On validation it keeps 68.2% of photos at 90.2% accuracy. On the test set, the same threshold keeps 69.0% of photos at 88.1% accuracy, and the photos below it are right only 46.2% of the time.

**Hardest classes** (`training/analyze_errors.py`, `training/results/error_analysis.json`), float16 test accuracy:

| Species | Accuracy | Most common wrong guess |
|---|---|---|
| Western Wood Pewee | 13% | Olive-sided Flycatcher |
| Brewer's Blackbird | 14% | Shiny Cowbird |
| Common Tern | 17% | Arctic Tern |
| American Crow | 23% | Fish Crow |
| California Gull | 30% | Herring Gull |
| Yellow-bellied Flycatcher | 34% | Acadian Flycatcher |
| Western Gull | 37% | Herring Gull |
| Bank Swallow | 40% | Gray Kingbird |
| Elegant Tern | 40% | Caspian Tern |
| Least Flycatcher | 45% | Acadian Flycatcher |

**Most confused pairs**, counting mistakes in both directions:

| Pair | Mix-ups |
|---|---|
| Long-tailed Jaeger and Pomarine Jaeger | 15 |
| Philadelphia Vireo and Warbling Vireo | 15 |
| Brandt's Cormorant and Pelagic Cormorant | 14 |
| Arctic Tern and Common Tern | 14 |
| Common Tern and Forster's Tern | 13 |
| Brewer's Blackbird and Shiny Cowbird | 12 |
| Acadian Flycatcher and Least Flycatcher | 12 |
| Herring Gull and Western Gull | 12 |
| Chuck-will's-widow and Whip-poor-will | 11 |
| American Crow and Fish Crow | 11 |

These are mostly pairs that birders also find hard from a single photo. `docs/misclassifications.png` shows the most confident mistake from each of the top pairs.

**Out-of-scope photos** (`training/evaluate_out_of_scope.py`, `training/results/out_of_scope.json`, figure `docs/out_of_scope_confidence.png`): I ran the shipped float16 file through the LiteRT interpreter, with the same preprocessing as `evaluate.py`, on two sets built by `training/scripts/build_out_of_scope.py`. The threshold stayed at 0.44. I didn't tune anything on these sets.

| Photo set | Source and licence | Photos | Below 0.44 ("Not confident") | Median top-1 |
|---|---|---|---|---|
| CUB test set (reference) | CUB-200-2011 | 5,794 | 31.0% | 0.64 |
| Birds of 20 species not in CUB, 15 each | iNaturalist research-grade observations, photos licensed CC0 or CC BY, one per observer | 300 | 88.7% | 0.17 |
| Photos with no bird: 20 classes of animals, plants, objects and scenes, 15 each | Open Images V7 validation set, CC BY 2.0, photos with any bird label skipped | 300 | 98.3% | 0.07 |

The bird species are American Robin, Mourning Dove, Black-capped Chickadee, Tufted Titmouse, House Finch, Great Blue Heron, Red-tailed Hawk, Bald Eagle, Canada Goose, Eastern Bluebird, European Robin, Eurasian Blue Tit, Common Chaffinch, Eurasian Magpie, Rainbow Lorikeet, Laughing Kookaburra, Common Myna, Red-whiskered Bulbul, Indian Peafowl and Superb Fairywren. All photos were shrunk to at most 512 px, like the app's photo picker does. The per-photo sources, licences and authors are in `training/out_of_scope/`, and the photos themselves are not in the repository.

On the non-bird set, 5 of 300 photos got a confident answer, all between 0.46 and 0.55: "Western Gull" for two road photos and a car, "Herring Gull" for a sea view and "Blue-winged Warbler" for a leaf. On the non-CUB birds, 34 of 300 (11.3%) got a confident wrong species, one of them above 0.9. House Finch (7 of 15 confident) and Rainbow Lorikeet (8 of 15) were the weakest, and the commonest confident answers were Painted Bunting (8) and Purple Finch (4). Many iNaturalist photos show small or distant birds, which also lowers confidence on in-scope species, so part of the 88.7% is down to harder photos rather than unknown species. These are small sets, so each rate has a margin of a few points.

**On-device check**: the app's Dart preprocessing plus tflite_flutter on the Android emulator matches the Python outputs to within 1.2e-6 on lossless inputs, and gives the same top-1 on all JPEG fixtures. Details are in `docs/emulator_results.md`.

## Known limitations

- **Closed set of 200 species.** CUB covers mostly North American birds. A bird from anywhere else, or a species missing from the list, still gets labelled as one of the 200, sometimes with high confidence. The "not confident" message helps but doesn't solve this, because the threshold was tuned on in-distribution validation photos only. On my out-of-scope set it caught 88.7% of non-CUB bird photos, so about 1 in 9 still got a confident wrong species.
- **Photo quality.** CUB photos are mostly clear shots with the bird filling a good part of the frame. Blurry, backlit, distant or partly hidden birds are much harder, and accuracy on those will be lower than the test numbers suggest.
- **Look-alike species.** Several CUB classes are hard even for people, such as some gulls, flycatchers, sparrows and vireos. These dominate the confusion list in the evaluation section.
- **Sex and age variation.** Females and juveniles of many species look very different from adult males, and CUB has fewer examples of them.
- **One bird per photo.** The model sees the whole image at 224x224, and there is no bird detector. With several birds, or a tiny bird in a big scene, it may focus on the wrong thing.

## Fairness of the evaluation

- All headline numbers come from the official CUB test split, which I never used for training, model selection, early stopping or picking the confidence threshold. Those decisions used my validation split, carved out of the training set.
- The accuracy of each TFLite file is measured by running that exact `.tflite` file through the LiteRT interpreter (`training/evaluate.py`), not the Keras model.
- CUB's train and test photos come from the same source and have a similar style, so the test score is an upper bound on what to expect from real phone photos.
- The backbone was pretrained on ImageNet, which also contains Flickr bird photos. Earlier work has pointed out that some CUB test images also appear in ImageNet. I haven't checked how much overlap there is, but if any, it would inflate the scores a little. This caveat applies to most published CUB results.
- I didn't tune hyperparameters against the test set. I report the single run I trained.
