# Collaborative Filtering with Naïve Bayes Classifier

This project implements a recommendation system using a Naïve Bayes Classifier approach to Collaborative Filtering (NBCF), based on the research paper "A Collaborative Filtering Approach Based on Naïve Bayes Classifier". 

## Overview

Collaborative filtering is a technique used in recommendation systems that makes predictions about a user's interests by collecting preferences from many users. This implementation uses a probabilistic approach with the Naïve Bayes classifier, combining both user-based and item-based collaborative filtering techniques.

## Datasets

The implementation supports working with different datasets:

1. **Toy Dataset**: A small example dataset built into the code for testing and demonstration purposes.
2. **FilmTrust Dataset**: A dataset containing user ratings for films, used for evaluation.
3. **MovieLens-1M**: A larger dataset with 1 million ratings from 6,000+ users on 4,000 movies.

## Algorithm Components

### IPO

![alt text](IPO.jpg)

### Prior Probability Calculation

Two types of prior probabilities are calculated:
- **User-based prior**: The probability of an item receiving a specific rating based on all previous ratings for that item.
- **Item-based prior**: The probability of a user giving a specific rating based on their previous rating patterns.

```python
def compute_priors(ratings, plausible_rating, alpha=0.01, R=8):
    # Calculates prior probabilities for both user-based and item-based approaches
    # with Laplace smoothing using alpha
```

### Likelihood Calculation

Two types of likelihoods are computed:
- **User-based likelihood**: Probability of a user giving a rating based on other users who rated the same items similarly.
- **Item-based likelihood**: Probability of an item receiving a rating based on other items that received similar ratings from the same user.

```python
def compute_likelihood_userbased(ratings, u, i, y, alpha=0.01, R=8):
    # Calculates the likelihood of user u giving rating y to item i
    
def compute_likelihood_itembased(ratings, u, i, y, alpha=0.01, R=8):
    # Calculates the likelihood of item i receiving rating y from user u
```

### Rating Prediction

The final rating prediction is calculated by combining the prior and likelihood probabilities from both user-based and item-based approaches:

```python
def predict_rating(ratings, u, i, prior_userbased, prior_itembased, plausible_rating, alpha=0.01, mode='hybrid'):
    # Predicts the rating for user u on item i using the specified mode (user, item, or hybrid)
```

Three prediction modes are available:
- `user`: Uses only user-based probabilities
- `item`: Uses only item-based probabilities
- `hybrid`: Combines both approaches, applying a weighting factor based on the number of available ratings

## Evaluation

The system is evaluated using standard recommendation system metrics:
- **RMSE (Root Mean Square Error)**: Measures the square root of the average squared differences between predicted and actual ratings
- **MAE (Mean Absolute Error)**: Measures the average absolute differences between predicted and actual ratings

The process involves:
1. Splitting the dataset into training and testing sets
2. Computing prior and likelihood probabilities on the training set
3. Predicting ratings for the test set
4. Calculating evaluation metrics (RMSE and MAE)

## Project Structure

- **finish training data real NEW.ipynb**: Main notebook with the implementation and evaluation of the NBCF approach
- **toy & real data.ipynb**: Contains experiments with both toy and real datasets
- **utils.py**: Utility functions for the implementation
- **film-trust/**: Directory containing the FilmTrust dataset
- **ml-1m/**: Directory containing the MovieLens-1M dataset
  - **ratings.dat**: User ratings
  - **movies.dat**: Movie information
  - **users.dat**: User information
  - **README**: Information about the dataset

## Usage

The code is implemented in Jupyter Notebooks and can be executed by running the cells in sequence. To experiment with different datasets, parameters, or approaches:

1. Load the dataset of choice
2. Split it into train and test sets
3. Compute priors and likelihoods
4. Make predictions
5. Evaluate results

Example workflow:
```python
# Load dataset
ratings_full, user_map, item_map, pR = load_filmtrust("./film-trust/ratings.txt")

# Split into train/test
ratings_train, test_set = train_test_split_matrix(ratings_full, test_ratio=0.2)

# Compute priors
prior_userbased, prior_itembased = compute_priors(ratings_train, pR)

# Predict and evaluate
y_true = []
y_pred = []
for u, i, actual in test_set:
    pred, _ = predict_rating(ratings_train, u, i, prior_userbased, prior_itembased, 
                            plausible_rating=pR, mode='hybrid')
    y_true.append(actual)
    y_pred.append(pred)

# Calculate metrics
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
```

## Implementation Details

### Smoothing

Laplace smoothing is applied to handle the cold start problem and zero probabilities. The code uses a small alpha value (0.01 by default) to add a small probability to all possible outcomes.

### Hybrid Mode

The hybrid approach combines user and item-based predictions using a weighted formula:
```
score_user = (prior_user * likelihood_user) ^ (1 / (1 + len_Iu))
score_item = (prior_item * likelihood_item) ^ (1 / (1 + len_Ui))
final_score = score_user * score_item
```

Where `len_Iu` is the number of items rated by user u, and `len_Ui` is the number of users who rated item i.

## Requirements

- Python 3.6+
- NumPy
- Pandas
- scikit-learn (for evaluation metrics)
- tqdm (for progress bars)

## Acknowledgements

This implementation is based on the paper:
"A Collaborative Filtering Approach Based on Naïve Bayes Classifier" 

The MovieLens dataset is provided by GroupLens Research at the University of Minnesota.
