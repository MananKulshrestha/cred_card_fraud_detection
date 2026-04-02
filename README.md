# Credit Card Fraud Detection: Handling Extreme Class Imbalance

This repository contains a machine learning project dedicated to identifying fraudulent credit card transactions. Because fraud detection datasets are typically plagued by extreme class imbalance, this project heavily emphasizes data preprocessing—specifically combining **SMOTE** (Synthetic Minority Over-sampling Technique) and **Random Undersampling**—to train robust and accurate predictive models.

---

## Dataset Overview

The project utilizes the widely recognized [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. 

* **Features:** Contains numerical input variables which are the result of a PCA transformation (V1 to V28), along with `Time` and `Amount`.
* **Target:** `Class` (0 for Normal transactions, 1 for Fraudulent transactions).
* **Challenge:** The original dataset is highly unbalanced, with frauds accounting for a tiny fraction of all transactions.

---

##  Data Preprocessing & Balancing Strategy

To prevent models from simply predicting the majority class, the dataset undergoes a rigorous preparation pipeline before training:

1.  **Feature Scaling:** The `Time` and `Amount` columns are scaled using `StandardScaler` to match the PCA-transformed features.
2.  **Strict Data Splitting:** The data is split into Train, Validation, and Test sets *before* any sampling techniques are applied. This prevents data leakage and ensures the validation/test sets reflect the true real-world distribution.
3.  **Hybrid Resampling (Training Data Only):**
    * **SMOTE:** Applied first to oversample the minority class (Fraud) until it reaches a user-defined percentage (e.g., 10%) of the majority class.
    * **Random Undersampling:** Applied next to reduce the majority class (Normal) until the final dataset reaches a balanced target ratio (e.g., 25% Fraud / 75% Normal).
4.  **Class Weights:** Computed for the finalized training set to pass into the neural networks and XGBoost algorithms.

---

## Models Developed

The notebook implements and compares four different algorithms to identify the best performer:

| Model | Architecture / Hyperparameters | Purpose |
| :--- | :--- | :--- |
| **Model 1: Sigmoid Baseline** | Single Dense layer with Sigmoid activation (Logistic Regression). | Establishes a baseline performance metric. |
| **Model 2: Basic Neural Network** | 2 Hidden Layers (16 units, 8 units) + Dropout (0.2). | Captures non-linear relationships in the data. |
| **Model 3: Deeper Neural Network** | 3 Hidden Layers (16, 8, 4 units) + Dropout (0.2, 0.3). | Explores if a deeper funnel structure improves feature extraction. |
| **Model 4: XGBoost** | `n_estimators=500`, `learning_rate=0.05`, `max_depth=5`, optimized `scale_pos_weight`. | State-of-the-art gradient boosting for tabular data. |

*(Note: Neural networks were trained using Early Stopping to prevent overfitting, restoring the best weights based on validation loss).*

---

## Results & Evaluation

The models were evaluated on the untouched **Test Set**. Because accuracy is a misleading metric for highly imbalanced data, **AUC (Area Under the ROC Curve)** and **Recall** were the primary focus. 

*Summary of Test Data Performance:*

* **Sigmoid Baseline:** Achieved an AUC of ~0.860.
* **Basic NN (2 Layers):** Achieved an excellent AUC of ~0.996, indicating strong separation capability.
* **Deeper NN (3 Layers):** Achieved an AUC of ~0.994, performing similarly to the shallower network.
* **XGBoost:** Achieved an Accuracy of **98.79%** and an AUC of **0.9701**. Crucially, XGBoost achieved a **Recall of 0.88** on the minority class, meaning it successfully caught 88% of all actual fraudulent transactions.

---

## How to Run

This project is built using Python and is designed to run in a Jupyter Notebook environment like Google Colab.

1.  **Clone the repository.**
2.  **Setup Kaggle API:** The notebook downloads the dataset directly from Kaggle. You will need your `kaggle.json` API token. 
    * When running the first cell, you will be prompted to upload your `kaggle.json` file.
    * The script will automatically move it to `~/.kaggle/` and download the dataset.
3.  **Install Dependencies:** Ensure you have the following libraries installed:
    * `pandas`, `numpy`, `matplotlib`, `seaborn`
    * `tensorflow`, `xgboost`, `scikit-learn`, `imbalanced-learn`
4.  **Run the Notebook:** Execute `smote&undersampling.ipynb` sequentially to preprocess the data, train the models, and view the confusion matrices.