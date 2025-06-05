# Collaborative Filtering with Naïve Bayes Classifier (NBCF)

This project implements a hybrid recommendation system using a Naïve Bayes Classifier approach to Collaborative Filtering, based on the research paper "A Collaborative Filtering Approach Based on Naïve Bayes Classifier". The implementation combines both user-based and item-based collaborative filtering techniques using probabilistic methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Algorithm Implementation](#algorithm-implementation)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Implementation Details](#implementation-details)

## 🎯 Overview

The Naïve Bayes Collaborative Filtering (NBCF) approach predicts user ratings by combining:
- **User-based CF**: Leveraging similar users' preferences
- **Item-based CF**: Utilizing similar items' rating patterns
- **Hybrid approach**: Combining both methods with weighted scoring

The system uses Bayesian probability to calculate the likelihood of a user giving a specific rating to an item, considering both user behavior patterns and item characteristics.

## 📊 Dataset

### FilmTrust Dataset
- **Location**: `./film-trust/`
- **Files**:
  - `ratings.txt`: User-item ratings (user_id, item_id, rating)
  - `train.txt`: Training set (80% of data)
  - `test.txt`: Test set (20% of data) 
  - `predictions.csv`: Model predictions vs ground truth
  - `trust.txt`: User trust relationships (not used in current implementation)

### Dataset Statistics
- **Users**: 1,508 unique users
- **Items**: 2,071 unique items  
- **Ratings**: 35,497 total ratings
- **Rating Scale**: 0.5 to 4.0 (8 possible values)
- **Train/Test Split**: 80/20 with random state 42

### Data Preprocessing
```python
# User and item mapping (0-indexed for matrix operations)
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}

# Example: Original user_id 470 → mapped to index 471
# This ensures consistent indexing between train/test sets
```

## 🔧 Algorithm Implementation

### 1. Prior Probability Calculation

Calculates the probability of each rating value occurring for users and items:

```python
def compute_priors(ratings, plausible_rating, alpha=1):
    """
    Computes prior probabilities with Laplace smoothing
    
    Returns:
    - prior_userbased: P(rating=y | item=j) for each item
    - prior_itembased: P(rating=y | user=u) for each user
    """
```

**Formula**: 
```
P(y|j) = (count_y + α) / (count_total + α × R)
```
Where R = number of possible rating values (8 in FilmTrust)

### 2. Likelihood Calculation

#### User-based Likelihood
```python
def compute_likelihood_userbased(ratings, u, i, y, alpha=0.01, R=8):
    """
    Calculates P(Ru,Iu | Ri=y) - probability of user u's ratings 
    given that item i receives rating y
    """
```

#### Item-based Likelihood  
```python
def compute_likelihood_itembased(ratings, u, i, y, alpha=0.01, R=8):
    """
    Calculates P(RUi,i | Ru=y) - probability of item i's ratings
    given that user u gives rating y
    """
```

### 3. Hybrid Prediction

```python
def predict_rating(ratings, u, i, prior_userbased, prior_itembased, plausible_rating, alpha=1):
    """
    Combines user-based and item-based approaches using weighted geometric mean
    
    Formula:
    score_user = (prior_user × likelihood_user)^(1/(1+|Iu|))
    score_item = (prior_item × likelihood_item)^(1/(1+|Ui|))  
    final_score = score_user × score_item
    """
```

**Key Features**:
- Uses `Decimal` precision for numerical stability
- Applies normalization based on data availability
- Returns the rating with highest probability score

## 📁 Project Structure

```
UAS/
├── FINAL MODEL.ipynb          # Main implementation notebook
├── README.md                  # This file
├── film-trust/               # Dataset directory
│   ├── ratings.txt           # Original ratings data
│   ├── train.txt            # Training set (auto-generated)
│   ├── test.txt             # Test set (auto-generated)
│   ├── predictions.csv      # Model predictions
│   ├── trust.txt            # Trust relationships
│   └── readme.txt           # Dataset description
└── generated_files/         # Exported matrices (CSV format)
    ├── ratings_full.csv
    ├── ratings_train.csv
    ├── prior_userbased.csv
    └── prior_itembased.csv
```

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install numpy pandas scikit-learn tqdm decimal
```

### Quick Start
1. **Open Jupyter Notebook**: [`FINAL MODEL.ipynb`](FINAL MODEL.ipynb)
2. **Run all cells sequentially** - the notebook includes:
   - Data loading and preprocessing
   - Train/test split (80/20)
   - Prior computation
   - Model training and prediction
   - Evaluation metrics calculation

### Key Code Sections

#### Data Loading
```python
# Load and create user-item matrix
ratings_full = load_filmtrust_full("./film-trust/ratings.txt")
```

#### Training
```python
# Compute priors from training data
prior_userbased, prior_itembased = compute_priors(ratings_train, pR)
```

#### Prediction
```python
# Predict ratings for test set
for u, i, actual in tqdm(test_set):
    pred, _ = predict_rating(ratings_train, u, i, prior_userbased, 
                           prior_itembased, plausible_rating=pR)
    y_true.append(actual)
    y_pred.append(pred)
```

## 📈 Evaluation Metrics

### Mean Absolute Error (MAE)

Two calculation methods implemented:

#### 1. Standard MAE (Library)
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

#### 2. User-based MAE (Paper Method)
```python
# Calculate MAE per user, then average across users
mae_total = 0
for user_id in user_data:
    mae_user = sum(abs(actual - predicted) for actual, predicted in user_ratings) / len(user_ratings)
    mae_total += mae_user
overall_mae = mae_total / len(user_data)
```

**Justification for Manual Calculation**: 
The manual calculation follows the paper's methodology where MAE is computed per user first, then averaged across all users. This approach gives equal weight to each user regardless of how many items they rated, which can provide different insights compared to the standard MAE calculation.

### Results Export
All predictions are saved to [`./film-trust/predictions.csv`](film-trust/predictions.csv) for detailed analysis:
```csv
y_true,y_pred
3.5,3.0
2.0,2.5
4.0,4.0
...
```

## 🎯 Results

The model achieves competitive performance on the FilmTrust dataset:
- **Test Set Size**: 7,100 predictions (20% of 35,497 ratings)
- **Rating Distribution**: Handles all 8 possible rating values (0.5-4.0)
- **Prediction Accuracy**: Evaluated using MAE metrics

## 🔍 Implementation Details

### Numerical Precision
- Uses `Decimal` class with 5-digit precision to handle small probabilities
- Prevents numerical underflow in likelihood calculations

### Optimization Features
- **Precomputed Priors**: Calculated once for entire test set
- **Efficient Filtering**: Only processes non-zero ratings
- **Progress Tracking**: Uses `tqdm` for long-running predictions

### Smoothing Parameters
- **Alpha (α)**: Laplace smoothing parameter
  - Prior calculation: α = 1 (default)
  - Likelihood calculation: α = 0.01 (prevents zero probabilities)
- **R**: Number of possible rating values (8 for FilmTrust)

### Memory Management
The notebook includes optional batch processing code (commented out) for handling larger datasets:
```python
# For very large datasets, use batch processing
# y_true, y_pred = predict_in_batches(test_set, ratings_train, 
#                                    prior_userbased, prior_itembased, 
#                                    pR, batch_size=500)
```

## 📚 References

- **Paper**: "A Collaborative Filtering Approach Based on Naïve Bayes Classifier"
- **Dataset**: FilmTrust dataset for movie recommendation research
- **Implementation**: Python with NumPy, Pandas, and scikit-learn

## 🔧 Technical Notes

### Data Mapping Consistency
The implementation ensures consistent user/item indexing between training and test sets:
- Original user ID 470 → Index 471 (0-based indexing)
- Original item ID 4 → Index 5 (0-based indexing)

This mapping is crucial for matrix operations and is preserved across train/test splits.

### Hybrid Weighting Strategy
The geometric mean approach with normalization factors provides adaptive weighting:
- Users with many ratings get less weight from user-based component
- Items with many ratings get less weight from item-based component
- Balances the influence of both approaches naturally

---

**Note**: This implementation focuses on the FilmTrust dataset. The code can be adapted for other rating datasets by modifying the data loading functions and adjusting the plausible rating values.