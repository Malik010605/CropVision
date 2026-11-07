CropVision: Machine Learning Approaches to Satellite Image Classification
Course: ITE351 — Artificial Intelligence and Applications
University: Hanyang University, Seoul Campus
Semester: Fall 2026
Project type: Machine Learning research project
Team Members
Name	Department	University	Email
Malik SAADI	School of Computer Science	Hanyang University (exchange)	malik.saadi@edu.ece.fr
Edgar PRADET	School of Computer Science	Hanyang University (exchange)	edgar.pradet@edu.ece.fr
Etienne SIOEN	School of Computer Science	Hanyang University (exchange)	etienne.sioen@edu.esilv.fr


The university affiliation shown above refers to the university hosting this course. Home-university affiliations can be added if required.

Table of Contents
- I. Introduction
- II. Datasets
- III. Methodology
- IV. Evaluation & Analysis
- V. Related Work
- VI. Conclusion & Discussion
- Video Presentation
- Project Status and Milestones
I. Introduction
Motivation
Satellite imagery can help us understand land use and observe environments such as forests, agricultural areas, residential neighborhoods, and bodies of water. Classifying these images manually at scale is time-consuming. Machine Learning (ML) and Deep Learning (DL) provide ways to automate land-use image classification by learning visual patterns from labeled examples.

In CropVision, we want to investigate how different approaches handle the same satellite image classification task. Instead of focusing only on obtaining a high accuracy score, we aim to understand the trade-offs between traditional ML methods and a convolutional neural network (CNN), including their strengths, limitations, and common errors.
Research Question
How do traditional Machine Learning methods and a Deep Learning approach compare in satellite image classification?
Objectives
1. Explore and describe a publicly available satellite image dataset.
2. Prepare a reproducible training, validation, and testing workflow.
3. Train and evaluate Random Forest and Support Vector Machine (SVM) classifiers.
4. Investigate Principal Component Analysis (PCA) as a dimensionality-reduction technique for traditional ML models.
5. Train and evaluate a Convolutional Neural Network (CNN).
6. Compare the approaches using the same held-out test set and appropriate metrics.
7. Discuss the results, misclassifications, and practical limitations of each approach.
II. Datasets
Planned Dataset: EuroSAT
We plan to use the RGB version of EuroSAT, a dataset of Sentinel-2 satellite image patches labeled by land-use or land-cover category. It contains 27,000 RGB images of 64 × 64 pixels, organized into 10 classes.
Class	Description
AnnualCrop	Areas with annual crops
Forest	Forested areas
HerbaceousVegetation	Areas dominated by herbaceous vegetation
Highway	Major roads and highways
Industrial	Industrial areas
Pasture	Pasture land
PermanentCrop	Areas with permanent crops
Residential	Residential areas
River	Rivers
SeaLake	Seas and lakes


Dataset source: EuroSAT — official GitHub repository

Original paper: Helber et al., EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
Planned Dataset Exploration
Once the dataset is downloaded, we will:
- Check the number of images in each class.
- Display representative examples from the 10 categories.
- Inspect image dimensions and pixel-value ranges.
- Identify potential sources of confusion between visually similar categories.
- Define training, validation, and test splits with a fixed random seed.
Current status: Dataset exploration has not started. Figures and observations will be added after inspection.
III. Methodology
Our experiments will follow a shared evaluation protocol so that the comparison between models is meaningful.
1. Data Preparation
We plan to load the RGB images, convert them into numerical representations, and create non-overlapping training, validation, and test sets. Preprocessing will be adapted to each model while keeping the test data unchanged across comparisons.

For traditional ML models, we will start with flattened pixel values as a baseline feature representation. We may standardize features where appropriate, fitting any transformations on the training set only to avoid data leakage.

For the CNN, we will use the original spatial image structure and normalize image values. Any data augmentation, if introduced, will be applied to the training set only.
2. Random Forest
Random Forest combines multiple decision trees to make predictions. It will serve as a traditional ML baseline and allow us to investigate how an ensemble of trees performs on image-derived features.
3. Support Vector Machine (SVM)
SVM is a supervised classification method that searches for decision boundaries between classes. We will evaluate an SVM on image-derived features, with feature scaling and hyperparameter selection performed using training and validation data.
4. Principal Component Analysis (PCA)
PCA is not a classification model. It is a dimensionality-reduction technique that transforms features into a smaller set of components capturing variation in the data.

We plan to compare selected traditional ML pipelines with and without PCA, where computationally feasible, to study how dimensionality reduction affects classification performance and training cost.
5. Convolutional Neural Network (CNN)
A CNN learns spatial features directly from images using convolutional layers. We plan to implement a CNN for the same 10-class classification task and compare it against the traditional ML baselines.

The final architecture, training configuration, and hyperparameters will be documented after implementation.
6. Experimental Protocol
To support a fair comparison, we plan to:
- Use consistent class labels and a common held-out test set.
- Choose preprocessing and hyperparameters without using test-set results.
- Record relevant training settings and random seeds.
- Evaluate all classifiers using the same metrics.
- Clearly document any differences in computational requirements.
Current status: The methodology is planned; no models have been trained yet.
IV. Evaluation & Analysis
We intend to compare our classifiers using the following metrics:
Metric	Purpose
Accuracy	Overall proportion of correctly classified images
Precision	How often predicted class labels are correct
Recall	How many images of each class are correctly identified
Macro F1-score	Balanced summary of precision and recall across classes
Confusion matrix	Identifies frequently confused categories
Training time	Provides context on computational cost


We will also inspect selected classification mistakes to better understand the strengths and weaknesses of each method.
Results
Not available yet. This section will contain tables, plots, confusion matrices, and comparisons based on completed experiments. We will not report expected or hypothetical scores as measured results.
V. Related Work
EuroSAT and Remote Sensing Classification
EuroSAT was introduced as a benchmark for land-use and land-cover classification using Sentinel-2 imagery. Its labeled image patches make it suitable for comparing different classification approaches under a shared dataset and evaluation procedure.

Our project will use EuroSAT as a starting point to examine the behavior of traditional ML classifiers and a CNN on the same task. A more detailed review of related studies and their experimental methods will be added during development.
Planned Tools and Libraries
- Python — primary programming language
- NumPy and Pandas — data handling and analysis
- Matplotlib — visualizations
- scikit-learn — Random Forest, SVM, PCA, and evaluation metrics
- PyTorch — CNN implementation and training
- Jupyter notebooks — experiments and reproducible documentation
- GitHub — code hosting and project blog
References
1. Helber, P., Bischke, B., Dengel, A., and Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. Dataset and publication information.
2. scikit-learn documentation.
3. PyTorch documentation.
Additional research papers and resources will be cited as they are used.
VI. Conclusion & Discussion
To be completed after the experiments.

This section will summarize the measured results, answer the research question, discuss limitations, and identify possible improvements or future experiments.
Video Presentation
Status: Not recorded yet.

A link to our video or audio presentation (approximately 10 minutes maximum) will be added here. It will explain the project motivation, dataset, methods, evaluation, and conclusions.
