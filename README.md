# Heart Disease Detection using Machine Learning

## Problem Statement
Heart disease is a leading cause of death worldwide, with Africa experiencing rising rates due to urbanization and lifestyle changes. This project implements machine learning classification models for early heart disease detection using clinical patient data. The heart.csv dataset contains 13 clinical parameters from 303 patients for binary classification of heart disease presence. We compare classical ML (Logistic Regression) with neural networks using various optimization techniques to determine the most effective approach.

## Dataset Overview

The heart.csv dataset is a comprehensive medical dataset containing  patient records with  clinical features for heart disease prediction. This dataset includes demographic information, clinical measurements, cardiac symptoms, diagnostic test results, and advanced cardiovascular indicators. The target variable indicates presence (1) or absence (0) of heart disease, with a balanced distribution of 54.5% positive cases and 45.5% negative cases.

### Exploring Heart Data using the "heart.csv" Dataset

**Introduction:**
In this notebook, we analyze data related to cardiac features of patients from the "heart.csv" dataset. This dataset provides comprehensive information about patients, including age, gender, blood pressure, cholesterol levels, electrocardiographic (ECG) features, and more.

**Dataset Information:**
The heart.csv dataset includes the following clinical features:

- **age**: The age of the patient (years)
- **sex**: Gender of the patient (0: female, 1: male)
- **cp**: Type of chest pain (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
- **trestbps**: Resting blood pressure (mm Hg on admission to the hospital)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1: true, 0: false)
- **restecg**: Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)
- **thalach**: Maximum heart rate achieved (beats per minute)
- **exang**: Exercise induced angina (1: yes, 0: no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia type (3: normal, 6: fixed defect, 7: reversable defect)
- **target**: Target variable indicating presence of heart disease (0: no disease, 1: disease present)

Each feature directly relates to cardiovascular health assessment and contributes to the classification task. The dataset contains 303 patient records with 13 input features and 1 target variable, making it suitable for binary classification of heart disease presence.

### Dataset Statistics and Analysis

**Data Overview:**
- **Total Records**: 303 patients
- **Features**: 13 clinical parameters
- **Target Variable**: Binary classification (0: no heart disease, 1: heart disease present)
- **Missing Values**: None (complete dataset)
- **Data Types**: Mixed (numerical and categorical)

**Target Distribution:**
- **Heart Disease Present**: 165 patients (54.5%)
- **No Heart Disease**: 138 patients (45.5%)
- **Class Balance**: Slightly imbalanced but manageable for binary classification

**Key Statistical Insights:**
- **Age Range**: 29-77 years (mean: 54.37 years)
- **Gender Distribution**: 207 males (68.3%), 96 females (31.7%)
- **Blood Pressure**: 94-200 mm Hg (mean: 131.62 mm Hg)
- **Cholesterol**: 126-564 mg/dl (mean: 246.26 mg/dl)
- **Max Heart Rate**: 71-202 bpm (mean: 149.65 bpm)

**Feature Categories:**
1. **Demographic Features**: age, sex
2. **Clinical Measurements**: trestbps, chol, fbs
3. **Cardiac Symptoms**: cp, exang
4. **Diagnostic Tests**: restecg, thalach, oldpeak
5. **Advanced Diagnostics**: slope, ca, thal

**Data Quality:**
- **Completeness**: 100% (no missing values)
- **Consistency**: All values within expected medical ranges
- **Reliability**: Clinical measurements from medical examinations
- **Scalability**: Features are on different scales, requiring normalization

**Preprocessing Requirements:**
- **Feature Scaling**: StandardScaler applied to numerical features
- **Categorical Encoding**: Ordinal encoding for categorical variables
- **Train-Test Split**: 80-20 split for model evaluation
- **Cross-Validation**: 5-fold cross-validation for hyperparameter tuning

## Model Implementation Results

### Training Instances Comparison Table

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Dropout | Accuracy | F1 Score | Recall | Precision | ROC AUC |
|-------------------|----------------|------------------|--------|----------------|------------------|---------------|---------|----------|----------|--------|-----------|---------|
| Instance 1 (Simple NN) | Default | None | 30 | No | 4 | 0.001 | 0.0 | 0.6957 | 0.7083 | 0.6538 | 0.7727 | 0.7019 |
| Instance 2 (Adam + L2) | Adam | L2 | 30 | Yes | 4 | 0.001 | 0.3 | 0.7609 | 0.7843 | 0.7692 | 0.8000 | 0.7596 |
| Instance 3 (RMSprop + L1) | RMSprop | L1 | 30 | Yes | 4 | 0.0005 | 0.4 | **0.8261** | **0.8519** | **0.8846** | **0.8214** | **0.8173** |
| Instance 4 (Adam + Dropout) | Adam | None | 30 | Yes | 4 | 0.0001 | 0.5 | 0.6739 | 0.7541 | 0.8846 | 0.6571 | 0.6423 |
| Instance 5 (Logistic Regression) | N/A | L2 (C=1) | N/A | N/A | N/A | N/A | N/A | 0.7609 | 0.7580 | 0.7596 | 0.7571 | N/A |
| Instance 6 (RMSprop + L2 + Dropout) | RMSprop | L2 | 30 | Yes | 4 | 0.001 | 0.2 | 0.7609 | 0.7843 | 0.7692 | 0.8000 | 0.7596 |

*Note: Results are based on test set performance. Early stopping was implemented with patience=10 and restore_best_weights=True. Best performing model highlighted in bold.*

## Discussion of Findings

### Optimization Techniques Analysis

**Instance 1 - Simple Neural Network (Baseline):**
- **Performance**: Achieved 69.57% accuracy with no optimization techniques
- **Analysis**: This baseline model demonstrates the fundamental learning capacity of neural networks but shows signs of overfitting due to lack of regularization
- **Loss Pattern**: Training accuracy reached 96.84% while test accuracy was only 69.57%, indicating significant overfitting
- **Key Insight**: Without regularization, the model memorizes training data rather than learning generalizable patterns

**Instance 2 - Adam Optimizer with L2 Regularization:**
- **Performance**: Improved to 76.09% accuracy with L2 regularization and dropout
- **Analysis**: L2 regularization (λ=0.01) effectively reduced overfitting by penalizing large weights
- **Key Insight**: The combination of Adam optimizer (adaptive learning rate) with L2 regularization provided better generalization
- **Dropout Impact**: 30% dropout rate helped prevent co-adaptation of neurons
- **Convergence**: Model showed stable training with validation accuracy reaching 91.11%

**Instance 3 - RMSprop Optimizer with L1 Regularization (BEST PERFORMING):**
- **Performance**: 82.61% accuracy, the best among all neural network models
- **Analysis**: L1 regularization (λ=0.01) induced sparsity by zeroing some weights, creating a more interpretable model
- **Optimizer Comparison**: RMSprop with lower learning rate (0.0005) showed excellent convergence
- **Dropout Strategy**: 40% dropout was optimal for this configuration
- **Key Success Factors**: The combination of L1 regularization with RMSprop created the most robust model

**Instance 4 - Adam Optimizer with High Dropout:**
- **Performance**: 67.39% accuracy, demonstrating the impact of aggressive regularization
- **Analysis**: 50% dropout rate was too aggressive, leading to underfitting
- **Learning Rate Impact**: Very low learning rate (0.0001) combined with high dropout caused slow learning
- **Trade-off**: Excessive regularization reduced model capacity too much

**Instance 5 - Logistic Regression (Classical ML):**
- **Performance**: 76.09% accuracy, outperforming most neural network models
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation tested:
  - **C values**: [0.001, 0.01, 0.1, 1, 10, 100] (L2 regularization strength)
  - **Solvers**: ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
  - **Max iterations**: [1000, 2000, 3000]
  - **Best parameters**: C=1.0, solver='lbfgs', max_iter=2000
- **Regularization**: L2 regularization (Ridge) with optimal strength C=1.0
- **Convergence**: LBFGS solver converged in 2000 iterations with stable coefficients
- **Feature Importance**: Coefficients provide direct interpretability of clinical features
- **Advantage**: Linear model captures the fundamental relationships in medical data effectively

**Instance 6 - RMSprop Optimizer with L2 Regularization and Dropout:**
- **Performance**: 76.09% accuracy, matching Adam + L2 performance
- **Analysis**: RMSprop with L2 regularization and 20% dropout provided balanced performance
- **Comparison**: Similar performance to Adam + L2, showing optimizer choice matters less with L2 regularization

### Critical Analysis of Optimization Techniques

**Regularization Impact:**
- **L1 regularization** proved most effective for this dataset, achieving 82.61% accuracy
- **L2 regularization** provided moderate improvement (76.09% vs 69.57% baseline)
- **High dropout (50%)** was counterproductive, causing underfitting
- **Optimal dropout** was 30-40% for this dataset size

**Optimizer Performance:**
- **RMSprop** with L1 regularization yielded the best results (82.61% accuracy)
- **Adam** performed well with L2 regularization but not as effectively as RMSprop with L1
- **Learning rate sensitivity**: Lower learning rates (0.0005) worked better with L1 regularization

**Early Stopping Effectiveness:**
- Early stopping with patience=10 prevented overfitting in all optimized models
- The technique was particularly valuable for models with regularization
- Validation loss monitoring helped identify optimal stopping points

**Model Architecture Insights:**
- 4-layer architecture (64→32→16→1) provided sufficient capacity
- The dataset size (303 samples) limited the benefits of very deep networks
- Feature scaling was crucial for model convergence

## Summary

### Best Combination Analysis

**Neural Network Optimization:**
The combination of **RMSprop optimizer + L1 regularization + 40% dropout + early stopping** (Instance 3) worked best for neural networks, achieving 82.61% accuracy. This combination provided:
- Optimal convergence through RMSprop's adaptive learning rate
- Effective feature selection through L1 regularization's sparsity
- Robust feature learning through moderate dropout
- Automatic training termination through early stopping

**Classical ML vs Neural Network Comparison:**

**Logistic Regression Advantages:**
- **Competitive Performance**: 76.09% accuracy vs 82.61% (best NN)
- **Faster Training**: Seconds vs minutes
- **Better Interpretability**: Clear feature importance and coefficients
- **Lower Computational Cost**: No GPU required
- **Hyperparameter Tuning**: GridSearchCV provided optimal C=1, solver='lbfgs'

**Neural Network Advantages:**
- **Higher Performance**: 82.61% accuracy (RMSprop + L1) vs 76.09% (LR)
- **Non-linear Relationships**: Can capture complex feature interactions
- **Feature Learning**: Automatic feature engineering through hidden layers
- **Flexibility**: Multiple optimization techniques available

**Recommendation:**
For this specific heart disease detection task, **RMSprop + L1 regularization** is the optimal choice due to:
1. Superior performance (82.61% vs 76.09% accuracy)
2. Effective feature selection through L1 regularization
3. Robust training through appropriate dropout
4. Good balance between model complexity and performance

The neural network with proper optimization significantly outperformed the classical algorithm, demonstrating the value of advanced optimization techniques when properly applied.

## Project Structure

```
heart_disease_detection/
├── Summative_Intro_to_ml_[Deolinda_Bogore]_assignment.ipynb  # Main notebook with all models
├── saved_models/                           # Trained models directory
│   ├── logistic_model.pkl                  # Logistic Regression model
│   ├── model_1_simple.h5                   # Simple neural network
│   ├── model_2_adam_l2.h5                  # Adam + L2 regularization
│   ├── model_3_rmsprop_l1.h5               # RMSprop + L1 regularization (BEST)
│   ├── model_4_adam_dropout.h5             # Adam + high dropout
│   ├── model_5_rmsprop_l2.h5               # RMSprop + L2 + dropout
│   └── model_architecture.png              # Neural network diagram
├── load_logistic_regression_model.py       # Demo script for model usage
├── requirements.txt                        # Python dependencies
├── heart.csv                              # Dataset
└── README.md                              # This file
```

## Running the Project

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Notebook:**
   ```bash
   jupyter notebook Summative_Intro_to_ml_[Deolinda_Bogore]_assignment.ipynb
   ```

3. **Load and Use Best Model:**
   ```bash
   python load_logistic_regression_model.py
   ```

## Model Loading Instructions

The best performing model (RMSprop + L1) can be loaded and used as follows:

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the best model
model = load_model('saved_models/model_3_rmsprop_l1.h5')

# Make predictions
predictions = model.predict(scaled_data)
binary_predictions = (predictions > 0.5).astype(int)
```

## Libraries Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (LogisticRegression, GridSearchCV, StandardScaler)
- **Deep Learning**: tensorflow, keras
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: pickle, joblib

## Key Findings

1. **Best Model**: RMSprop + L1 regularization achieved 82.61% accuracy
2. **Optimization Impact**: Proper regularization improved performance by 13% over baseline
3. **Regularization Strategy**: L1 regularization was more effective than L2 for this dataset
4. **Dropout Sensitivity**: Moderate dropout (30-40%) optimal, high dropout (50%) harmful
5. **Optimizer Choice**: RMSprop outperformed Adam when combined with L1 regularization
6. **Model Comparison**: Neural networks with optimization outperformed classical ML by 6.52%

## Technical Achievements

✅ **Implemented 6 distinct models** with different optimization techniques  
✅ **Achieved 82.61% accuracy** through proper hyperparameter tuning  
✅ **Demonstrated understanding** of regularization, dropout, and optimizers  
✅ **Created modular, reusable code** with comprehensive evaluation  
✅ **Successfully saved and loaded models** for deployment  
✅ **Comprehensive error analysis** with all required metrics  

## Learning Outcomes

• **Mastered neural network optimization techniques**  
• **Understood the impact of different regularization methods**  
• **Learned to compare classical ML vs deep learning approaches**  
• **Developed skills in hyperparameter tuning and model evaluation**  
• **Gained practical experience in medical ML applications**  

## Author

**Deolinda Bio Bogore**  
African Leadership University  
BSE - Introduction to Machine Learning 
