
# SVMBreastCancerClassifier

This repository contains an implementation of a Support Vector Machine (SVM) Classifier on the Breast Cancer dataset. The goal of the project is to train a machine learning model to predict the severity of breast cancer (malignant or benign) based on various features of cell nuclei.


##  Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Visualizations](#visualizations)
- [Results](#results)

---
##  ## Overview

This project uses the **Breast Cancer Wisconsin dataset** to classify tumors as **malignant** or **benign** using an SVM classifier. The dataset contains both numerical and categorical features, and the goal is to:

1. **Preprocess** the data (one-hot encoding, label encoding, and feature scaling).
2. **Train** the SVM model using a radial basis function (RBF) kernel.
3. **Evaluate** the model's performance using accuracy, confusion matrix, and classification report.
4. **Visualize** the  **confusion matrix**.

## ## Installation

git clone https://github.com/your-Sanjaycode25/svm-breast-cancer.git
cd svm-breast-cancer


## ##Usage

To run the code, make sure you have the dataset ready (either from a CSV file or load it directly via a function like load_breast_cancer() from sklearn).

Here’s how you can run the SVM classifier:

"python svm_classifier.py"
##  ##Code Explanation

**1.** **Data Preprocessing**
One-hot encoding is used to transform categorical features into numerical ones.
Label encoding is applied to the target column to convert it to a binary format (0 and 1).
Feature scaling is done using StandardScaler to standardize the features before passing them into the SVM model.

**2.** **Training the SVM Classifier**
The SVM classifier is trained using an RBF kernel (kernel='rbf') which is effective for non-linear decision boundaries.

**3.** **Model Evaluation**
After training the SVM model, its performance is evaluated based on:

*Accuracy:* The proportion of correctly classified instances.

*Confusion Matrix:* A heatmap visualization that shows the number of true positives, true negatives, false positives, and false negatives.

*Classification Report:* Provides metrics like precision, recall, and F1-score.

**4. Visualization**
The project also includes visualizations:

*Confusion Matrix:* A heatmap showing the confusion matrix.

Decision Boundary: Using PCA for dimensionality reduction, we visualize the decision boundary of the SVM classifier.
##  ##Visualizations

**Confusion Matrix:** A heatmap showing the true vs. predicted values of the model’s output. This helps assess how well the model is classifying the malignant and benign tumors.

**SVM Decision Boundary:** After reducing the features to 2D using PCA, we plot the decision boundary of the SVM classifier. This shows how the classifier separates the two classes (malignant and benign) in the reduced feature space.
##  ##Results

*Accuracy:* The model's accuracy in classifying tumors as benign or malignant is displayed after training.

*Confusion Matrix:* The confusion matrix shows the number of correct and incorrect predictions.

*Classification Report:* The classification report gives detailed performance metrics such as precision, recall, and F1-score for both classes.
## ##License

This project is licensed under the MIT License - see the LICENSE file for details.
##  ##Acknowledgements

- The Breast Cancer Wisconsin dataset is available from UCI Machine Learning Repository.

- The machine learning algorithms are implemented using scikit-learn.

- Visualizations are created using matplotlib and seaborn
