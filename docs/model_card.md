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

<!-- Filled from training/results/train_summary.json and the training logs. -->

## Evaluation

<!-- Filled from training/results/*.json. -->

## Known limitations

- **Closed set of 200 species.** CUB covers mostly North American birds. A bird from anywhere else, or a species missing from the list, still gets labelled as one of the 200, sometimes with high confidence. The "not confident" message helps but doesn't solve this, because the threshold was tuned on in-distribution validation photos only.
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
