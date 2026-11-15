# CropVision Presentation Script

Target duration: **9 minutes 30 seconds**. The three sections are designed for about **123 spoken words per minute**. Pause briefly when a chart first appears, but keep the handoffs at 3:10 and 6:20.

## Edgar PRADET — 0:00–3:10

### Slide 1 — CropVision (0:00–0:25)

Hello, we are Edgar Pradet, Malik Saadi, and Etienne Sioen. This is CropVision, our project for ITE351, Artificial Intelligence and Applications at Hanyang University. We study satellite image classification and ask one question: how do traditional machine-learning methods compare with a compact deep-learning approach when every model receives the same images?

### Slide 2 — Motivation and fair comparison (0:25–1:05)

Satellite images can support land-use and land-cover analysis, but visual inspection does not scale well to large image collections. We therefore compare two model families. The first uses flattened RGB pixels with classical classifiers. The second is a convolutional neural network, or CNN, which keeps the two-dimensional image structure. Our objective is not to claim one method is always best. It is to make a controlled comparison: identical image identities, one fixed random seed, model-appropriate preprocessing, and the same evaluation metrics.

### Slide 3 — EuroSAT RGB (1:05–2:05)

We use the RGB version of EuroSAT. It contains 27,000 verified Sentinel-2 image patches. Every image has three visible-colour channels and a resolution of 64 by 64 pixels. The dataset has ten classes, including forest, river, highway, residential areas, industrial areas, pasture, permanent crops, and Sea or Lake. The chart shows that the classes are reasonably balanced, although their sizes range from 2,000 to 3,000 images. We use RGB only, not the thirteen-band multispectral release. This makes the input easy to reproduce and keeps the experiments manageable on an eight-gigabyte Apple Silicon Mac, but it also limits the spectral information available to the models.

### Slide 4 — Split integrity and leakage prevention (2:05–3:10)

We created one deterministic, stratified split with seed 42. Seventy percent, or 18,900 images, are used for training. Fifteen percent, or 4,050 images, form the validation set, and the remaining 4,050 images form the final test set. Stratification keeps class proportions similar across all three subsets. The manifests contain all 27,000 unique images, with no overlap.

Preventing leakage is essential because information from validation or test images could make performance look better than it really is. For the classical pipelines, scaling and PCA are fitted only on training data. Validation data is used to select configurations and the CNN checkpoint. The test set remains untouched until one final evaluation of the frozen models. This gives us a fair estimate on unseen samples from the same dataset split. This separation makes every reported stage easier to audit and reproduce. I will now hand over to Malik, who will explain the models and training process.

## Malik SAADI — 3:10–6:20

### Slide 5 — Traditional machine-learning pipelines (3:10–4:25)

We evaluated three classical baselines. First, Random Forest receives each image as 12,288 flattened pixel values: 64 times 64 times three channels. It combines 300 decision trees and does not require learned feature scaling.

The second pipeline adds standardization and principal component analysis, or PCA. PCA is not a classifier. It compresses the 12,288 pixel features into 128 components fitted only from the training set. A 300-tree Random Forest then uses this lower-dimensional representation.

The third pipeline applies the same leakage-safe scaling and 128-component PCA before an RBF-kernel support vector machine. The selected SVM uses C equal to 10 and gamma set to scale. We did not train a full raw-pixel RBF SVM. Its kernel computation and possible support-vector memory use were not safely bounded for our 18,900 training samples on an eight-gigabyte Mac. SVM plus PCA is therefore our scientifically defensible, resource-limited baseline.

### Slide 6 — Compact CNN architecture (4:25–5:30)

Our deep-learning model is a custom CNN with 620,362 trainable parameters. The RGB input passes through three convolution blocks with 32, then 64, then 128 output channels. Each block contains a three-by-three convolution, a ReLU activation, and two-by-two max pooling. The trained model does not use batch normalization. After the third block, adaptive average pooling produces 128 feature maps of four by four pixels, or 2,048 values.

The classification head flattens those values, applies a fully connected layer with 256 hidden units, ReLU, and 30 percent dropout, then outputs ten class scores. This architecture is intentionally simple and small enough for our memory limit. Unlike flattened-pixel models, convolution preserves local neighbourhoods and reuses the same filters across the image.

### Slide 7 — Training and model selection (5:30–6:20)

We trained with cross-entropy loss and Adam, using a learning rate of 0.001, weight decay of 0.0001, and batches of 64. The maximum was 30 epochs, with early stopping patience of five. After every epoch, we recorded training and validation loss, accuracy, and macro-F1. Checkpoint selection used validation macro-F1 only; no test image was loaded during training.

The best checkpoint was epoch 21, with 93.36 percent validation accuracy and 0.9318 validation macro-F1. Training continued to epoch 26 before early stopping. The curves show training still improving while validation fluctuates, which indicates some overfitting after the best epoch. Saving the validation-selected checkpoint prevents those later epochs from replacing the best model. The same seed controls initialization. Etienne will now present the one-shot test results and our conclusion.

## Etienne SIOEN — 6:20–9:30

### Slide 8 — Final held-out test results (6:20–7:25)

After freezing every pipeline and the selected CNN checkpoint, we evaluated all four models once on the same 4,050 held-out test images. These are test results, not validation results.

Random Forest achieved 68.47 percent accuracy and 0.6695 macro-F1. Random Forest plus PCA reached 67.90 percent and 0.6599. SVM plus PCA reached 67.23 percent and 0.6632. The CNN achieved 93.88 percent test accuracy and 0.9369 macro-F1.

The CNN result is about 25 percentage points above the classical baselines in accuracy. This gap is consistent with convolutional features preserving local spatial structure, while the baseline models use flattened pixels or compressed linear components. However, our experiment does not isolate a single cause, so we present this as an observed association rather than proof. The table also shows inference time for the complete test set: around 0.20 to 0.25 seconds for both forests, 3.45 seconds for SVM plus PCA, and 3.52 seconds for the CNN in our recorded evaluation environment.

### Slide 9 — Class-level errors (7:25–8:20)

Macro-F1 gives every class equal importance. For the plain Random Forest, Highway and Permanent Crop were especially difficult, with F1 scores of 0.3882 and 0.4690. In contrast, every CNN class had an F1 score above 0.90.

The CNN confusion matrix still reveals errors. Nineteen Forest images were predicted as Pasture. Nineteen Permanent Crop images were predicted as Herbaceous Vegetation, and fourteen Highway images were predicted as River. These confusions are plausible visual similarities in small RGB patches, but we did not perform a causal study. The point is that the CNN reduced errors across all classes.

### Slide 10 — Conclusion and limitations (8:20–9:30)

Our answer to the research question is clear for this controlled experiment: the compact CNN generalized much better than the three traditional flattened-pixel baselines. PCA reduced the feature dimension and made the RBF SVM feasible, but it did not improve the Random Forest test score.

Several limitations restrict broader conclusions. We used RGB rather than multispectral bands, one seed and one stratified split, and no geographic separation between sites. We did not estimate confidence intervals, and computational limits kept the hyperparameter search small and excluded raw-pixel RBF SVM. Finally, classifying isolated patches is not the same as producing a complete land-cover map.

Future work should evaluate repeated and geographically separated splits, multispectral inputs, stronger data augmentation, and carefully controlled architecture comparisons. The code, configurations, and reproducibility instructions are available in our GitHub repository. Thank you for listening.

## Timing summary

| Speaker | Slides | Scheduled time | Target words | Estimated speaking rate |
|---|---:|---:|---:|---:|
| Edgar PRADET | 1–4 | 0:00–3:10 | approximately 390 | approximately 123 words/minute |
| Malik SAADI | 5–7 | 3:10–6:20 | approximately 390 | approximately 123 words/minute |
| Etienne SIOEN | 8–10 | 6:20–9:30 | approximately 390 | approximately 123 words/minute |
| **Total** | **1–10** | **9:30** | **approximately 1,170** | **approximately 123 words/minute** |

