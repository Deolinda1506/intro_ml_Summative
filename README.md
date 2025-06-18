# Heart Disease Detection using Machine Learning

## Problem Statement
Heart disease is a leading cause of death worldwide, with Africa experiencing rising rates due to urbanization and lifestyle changes. This project implements machine learning classification models for early heart disease detection using clinical patient data. The heart.csv dataset contains 13 clinical parameters from 303 patients for binary classification of heart disease presence. We compare classical ML (Logistic Regression) with neural networks using various optimization techniques to determine the most effective approach.

## Dataset Overview
The heart.csv dataset contains  patient records with clinical features including age, sex, chest pain type, blood pressure, cholesterol, blood sugar, ECG results, heart rate, exercise angina, ST depression, slope, vessel count, and thalassemia type. The target variable indicates heart disease presence (1) or absence (0), with 54.5% positive cases and 45.5% negative cases.

## Discussion of Findings

### Training Instances Comparison Table

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Learning Rate | Dropout | Accuracy | F1 Score | Recall | Precision | ROC AUC |
|-------------------|----------------|------------------|--------|----------------|---------------|---------|----------|----------|--------|-----------|---------|
| Instance 1 (Simple NN) | Default (Adam) | None | 30 | No | 0.001 | 0.0 | 0.6957 | 0.7083 | 0.6538 | 0.7727 | 0.7019 |
| Instance 2 (Adam + L2) | Adam | L2 | 30 | Yes | 0.001 | 0.3 | 0.7609 | 0.7843 | 0.7692 | 0.8000 | 0.7596 |
| Instance 3 (RMSprop + L1) | RMSprop | L1 | 30 | Yes | 0.0005 | 0.4 | **0.8261** | **0.8519** | **0.8846** | **0.8214** | **0.8173** |
| Instance 4 (Adam + Dropout) | Adam | None | 30 | Yes | 0.0001 | 0.5 | 0.6739 | 0.7541 | 0.8846 | 0.6571 | 0.6423 |
| Instance 5 (Logistic Regression) | N/A | L2 (C=1) | N/A | N/A | N/A | N/A | 0.7609 | 0.7580 | 0.7596 | 0.7571 | N/A |
| Instance 6 (RMSprop + L2 + Dropout) | RMSprop | L2 | 30 | Yes | 0.001 | 0.2 | 0.7609 | 0.7843 | 0.7692 | 0.8000 | 0.7596 |

*Note: Results are based on test set performance. Early stopping was implemented with patience=10. Best performing model highlighted in bold.*

### Analysis of Optimization Techniques

**Instance 1 - Simple Neural Network (Baseline):**
Model one had everything set to defaults (learning rate of 0.001 for Adam) with no optimization or techniques to improve the model used. It achieved an accuracy of 0.6957 which is the lowest among all models due to severe overfitting. Due to having no early stopping, dropout, or regularization, it went past its minima and kept memorizing data which is evident by the huge gap between training and validation performance with training accuracy reaching 96.84% while test accuracy was only 69.57%. As for the error metrics, the precision of 0.7727 and recall of 0.6538 indicate poor generalization, with the model showing high false positive rates. The F1 score of 0.7083 reflects the imbalance between precision and recall. Due to how overfitted it is, I wouldn't trust it with outside data as it would probably not be able to generalize to new patients.

**Instance 2 - Adam Optimizer with L2 Regularization:**
Model two implemented L2 regularization with a strength of λ=0.01 and 30% dropout rate, along with early stopping to prevent overfitting. It achieved an accuracy of 0.7609, showing significant improvement over the baseline model. The L2 regularization effectively penalized large weights, preventing the model from memorizing training data. Early stopping with patience=10 ensured the model stopped training when validation loss started increasing. The precision of 0.8000 and recall of 0.7692 show much better balance than the baseline, with an F1 score of 0.7843 indicating good overall performance. The ROC AUC of 0.7596 demonstrates decent discriminative ability. This model shows good generalization potential and would be more reliable for real-world applications.

**Instance 3 - RMSprop Optimizer with L1 Regularization (BEST PERFORMING):**
Model three used RMSprop optimizer with L1 regularization (λ=0.01), 40% dropout, and early stopping, achieving the best performance with 82.61% accuracy. The L1 regularization induced sparsity by zeroing some weights, creating a more interpretable model while effectively preventing overfitting. RMSprop with a lower learning rate of 0.0005 provided excellent convergence, adapting the learning rate for each parameter. The 40% dropout rate was optimal for this configuration, preventing co-adaptation of neurons without being too aggressive. The precision of 0.8214 and recall of 0.8846 show excellent balance, with the highest F1 score of 0.8519 among all models. The ROC AUC of 0.8173 indicates strong discriminative ability. This combination created the most robust and generalizable model, making it the optimal choice for heart disease detection.

**Instance 4 - Adam Optimizer with High Dropout:**
Model four implemented aggressive regularization with 50% dropout rate and a very low learning rate of 0.0001, but achieved only 67.39% accuracy, demonstrating the negative impact of excessive regularization. The high dropout rate was too aggressive, leading to underfitting as the model lost too much information during training. The very low learning rate combined with high dropout caused slow learning and prevented the model from reaching optimal performance. Despite having a high recall of 0.8846, the precision of 0.6571 was poor, resulting in many false positives. The F1 score of 0.7541 and ROC AUC of 0.6423 reflect the model's poor discriminative ability. This instance clearly shows that more regularization is not always better and that finding the right balance is crucial for optimal performance.

**Instance 5 - Logistic Regression (Classical ML):**
Model five used Logistic Regression with L2 regularization, achieving 76.09% accuracy through extensive hyperparameter optimization using GridSearchCV with 5-fold cross-validation. The optimization tested various C values [0.001, 0.01, 0.1, 1, 10, 100], solvers ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], and max iterations [1000, 2000, 3000], ultimately selecting C=1.0, solver='lbfgs', max_iter=2000 as optimal parameters. The L2 regularization with C=1.0 provided the right amount of regularization strength. The LBFGS solver converged in 2000 iterations with stable coefficients, indicating good numerical stability. The precision of 0.7571 and recall of 0.7596 show excellent balance, with an F1 score of 0.7580. This model provides direct interpretability through feature coefficients and demonstrates that classical ML can be highly effective for medical diagnosis tasks, especially when properly tuned.

**Instance 6 - RMSprop Optimizer with L2 Regularization and Dropout:**
Model six combined RMSprop optimizer with L2 regularization (λ=0.01) and 20% dropout, achieving 76.09% accuracy, matching the performance of Instance 2 (Adam + L2). The L2 regularization provided the same weight penalty as Instance 2, while the 20% dropout rate was more conservative than the 30% used in Instance 2. The precision of 0.8000 and recall of 0.7692 are identical to Instance 2, with the same F1 score of 0.7843 and ROC AUC of 0.7596. This similarity in performance suggests that when using L2 regularization, the choice between Adam and RMSprop optimizers has minimal impact on final results. The model shows good generalization with balanced error metrics, but the performance is not as strong as the L1 regularization approach used in Instance 3, indicating that L1 regularization was more effective for this specific dataset and task.


## Summary

### Best Combination Analysis

**Neural Network Optimization:**
The optimal combination for neural networks was **RMSprop optimizer + L1 regularization + 40% dropout + early stopping** (Instance 3), achieving the highest accuracy of 82.61% among all models. This combination succeeded through several synergistic effects: RMSprop's adaptive learning rate of 0.0005 provided excellent convergence by adjusting learning rates for each parameter individually, while L1 regularization with λ=0.01 induced sparsity by zeroing some weights, creating a more interpretable model that effectively prevented overfitting. The 40% dropout rate was optimal for this configuration, preventing co-adaptation of neurons without being too aggressive like the 50% dropout that caused underfitting in Instance 4. Early stopping with patience=10 ensured the model stopped training at the optimal point, preventing overfitting while maximizing performance. The resulting model showed excellent balance with precision of 0.8214, recall of 0.8846, and F1 score of 0.8519, making it the most reliable choice for real-world heart disease detection applications.

### Classical ML vs Neural Network Comparison

**Model Comparison Analysis:**
The comparison between Logistic Regression (76.09% accuracy) and the best neural network (82.61% accuracy) reveals key trade-offs between classical and deep learning approaches. Logistic Regression achieved its performance through extensive GridSearchCV optimization (C=1.0, solver='lbfgs', max_iter=2000) and offers significant advantages in training speed (seconds vs minutes), computational efficiency (no GPU required), and interpretability through direct feature coefficients. However, the neural network's 6.52% accuracy improvement demonstrates superior capability in capturing non-linear relationships and complex feature interactions crucial for medical diagnosis. The RMSprop + L1 regularization combination provided the best balance of performance and interpretability, with L1 regularization's sparsity-inducing properties making the neural network more interpretable than typical deep learning models. While Logistic Regression remains excellent for resource-constrained environments, the neural network's superior performance makes it the optimal choice for real-world heart disease detection where accuracy is paramount. The 6.52% improvement represents a significant clinical enhancement, justifying the increased complexity and computational requirements for medical applications.

### Recommendation

**Primary Recommendation:**
For heart disease detection applications, implement the **RMSprop + L1 regularization neural network** (Instance 3) as the primary model. This configuration achieved 82.61% accuracy with excellent balance between precision (0.8214) and recall (0.8846), making it the most reliable choice for clinical deployment. The L1 regularization's feature selection capabilities provide interpretability crucial for medical applications, while the 40% dropout and early stopping ensure robust generalization to new patient data.



**Alternative Recommendation:**
For resource-constrained environments or rapid prototyping, use **Logistic Regression** with L2 regularization (C=1.0, solver='lbfgs'). While achieving lower accuracy (76.09%), it provides faster training, better interpretability, and requires minimal computational resources. This model serves as an excellent baseline and can be deployed in settings where computational efficiency is prioritized over maximum accuracy.



## Project Structure

```
intro_ml_Summative/
├── Summative_Intro_to_ml_[Deolinda_Bogore]_assignment.ipynb  # Main notebook with all models
├── saved_models/                           # Trained models directory
├── requirements.txt                        # Python dependencies
├── heart.csv                              # Dataset
└── README.md                              # This file
```

## Running the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Deolinda1506/intro_ml_Summative.git
   cd intro_ml_Summative
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook Summative_Intro_to_ml_[Deolinda_Bogore]_assignment.ipynb
   ```

## Loading and Using Models

### Load Best Neural Network Model (RMSprop + L1)

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the best performing model
model = load_model('saved_models/model_3_rmsprop_l1.h5')

# Make predictions
predictions = model.predict(scaled_data)
binary_predictions = (predictions > 0.5).astype(int)
```

### Model Performance

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| RMSprop + L1 (Best) | 82.61% | 0.8519 | 0.8214 | 0.8846 |

## Author

**Deolinda Bio Bogore**  
**Video Presentation**
https://drive.google.com/file/d/123iDKq1x48wc0zgBUX8OOYJrkVhLo0FH/view?usp=sharing

African Leadership University  
BSE - Introduction to Machine Learning 
