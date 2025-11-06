# Diabetes Prediction (women) – Machine Learning Practice Project

This project uses **Machine Learning** to predict whether a person has **diabetes** based on medical test data.
It’s built as a **practice project** to understand data preprocessing, scaling, training, and prediction using **Support Vector Machine (SVM)**.

---

## Project Overview

* **Goal:** Predict if a person is diabetic or not
* **Algorithm Used:** Support Vector Machine (SVM)
* **Dataset:** PIMA Indians Diabetes Dataset
* **Language:** Python
* **Libraries Used:**

  * `pandas` – for loading and managing the dataset
  * `numpy` – for handling arrays and numerical data
  * `scikit-learn` – for model building, scaling, and accuracy testing

---

## Steps and Workflow

### 1. Importing Required Libraries

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2. Loading the Dataset

```python
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
```

Check dataset structure, count outcomes, and calculate mean values for each class.

### 3. Separating Data and Labels

```python
data = diabetes_dataset.drop(columns='Outcome', axis=1)
result = diabetes_dataset['Outcome']
```

### 4. Data Standardization

```python
scaler = StandardScaler()
scaler.fit(data)
standardized_data = scaler.transform(data)
```

 Standardization ensures all feature values are in the same range, helping the model learn effectively.

### 5. Splitting Dataset

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    standardized_data, result, test_size=0.2, stratify=result, random_state=2)
```

80% of data is used for training and 20% for testing.

### 6. Model Training

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 7. Model Evaluation

```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
```

### 8. Making Predictions

```python
input_data = (1,189,60,23,846,30.1,0.398,59)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = classifier.predict(std_data)
```

If prediction = 0 → Not Diabetic
If prediction = 1 → Diabetic

---

## What I Learned

* How to preprocess data using Pandas and NumPy
* The importance of data standardization using `StandardScaler`
* How to train and evaluate an **SVM** classifier
* Creating a simple predictive system for new inputs

---

## Future Improvements

* Add GUI or web interface for user input
* Compare SVM with Logistic Regression or Random Forest
* Visualize data correlations and feature importance

---

### Author

**Aruna Madugeeth**
Machine Learning Practice Project | SVM | Diabetes Detection | Python | Scikit-learn
