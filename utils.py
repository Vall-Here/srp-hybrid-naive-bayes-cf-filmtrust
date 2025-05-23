# import numpy as np

# ratings = np.array([
#     [0, 1, 2, 2, 5, 0, 4, 3, 5], 
#     [1, 5, 3, 0, 2, 3, 4, 3, 0], 
#     [1, 1, 2, 0, 2, 4, 4, 5, 0], 
#     [3, 2, 2, 3, 0, 1, 3, 2, 0], 
#     [5, 1, 5, 5, 4, 4, 5, 2, 0], 
# ])


# def compute_priors(ratings, alpha=0.01, R=5):
#     num_users = len(ratings)
#     num_items = len(ratings[0])
#     rating_values = list(range(1, R + 1))

#     prior_userbased = [[0 for _ in range(num_items)] for _ in rating_values]
#     prior_itembased = [[0 for _ in range(num_users)] for _ in rating_values]

#     for y in rating_values:
#         y_index = y - 1

#         # Prior user-based (per item j)
#         for j in range(num_items):
#             count_y = 0
#             count_nonzero = 0
#             for u in range(num_users):
#                 r = ratings[u][j]
#                 if r != 0:
#                     count_nonzero += 1
#                     if r == y:
#                         count_y += 1
#             prior_userbased[y_index][j] = (count_y + alpha) / (count_nonzero + alpha * R)

#         # Prior item-based (per user u)
#         for u in range(num_users):
#             count_y = 0
#             count_nonzero = 0
#             for j in range(num_items):
#                 r = ratings[u][j]
#                 if r != 0:
#                     count_nonzero += 1
#                     if r == y:
#                         count_y += 1
#             prior_itembased[y_index][u] = (count_y + alpha) / (count_nonzero + alpha * R)

#     return prior_userbased, prior_itembased

# def compute_likelihood_userbased(ratings, u, i, y, alpha=0.01, R=5):
#     num_users = len(ratings)
#     num_items = len(ratings[0])
#     Iu = [j for j in range(num_items) if j != i and ratings[u][j] != 0]
#     product = 1.0

#     for j in Iu:
#         k = ratings[u][j]
#         count_joint = 0
#         count_cond = 0
#         for v in range(num_users):
#             if ratings[v][i] == y:
#                 if ratings[v][j] != 0:
#                     count_cond += 1
#                     if ratings[v][j] == k:
#                         count_joint += 1
#         prob = (count_joint + alpha) / (count_cond + alpha * R)
#         print(prob)
#         product *= prob
#     print("======")
#     print(product, end="\n\n")

#     return product

# def compute_likelihood_itembased(ratings, u, i, y, alpha=0.01, R=5):
#     num_users = len(ratings)
#     num_items = len(ratings[0])
#     Ui = [v for v in range(num_users) if v != u and ratings[v][i] != 0]
#     product = 1.0

#     for v in Ui:
#         k = ratings[v][i]
#         count_joint = 0
#         count_cond = 0
#         for j in range(num_items):
#             if ratings[u][j] == y:
#                 if ratings[v][j] != 0:
#                     count_cond += 1
#                     if ratings[v][j] == k:
#                         count_joint += 1
#         prob = (count_joint + alpha) / (count_cond + alpha * R)
#         product *= prob

#     return product


# def predict_rating(ratings, u, i, prior_userbased, prior_itembased, alpha=0.01, R=5, mode='hybrid'):
#     scores = []
#     all_likelihood_user = []  
#     all_likelihood_item = []
#     all_combined = []  

#     for y in range(1, R + 1):
#         y_index = y - 1
#         prior_user = prior_userbased[y_index][i]
#         prior_item = prior_itembased[y_index][u]

#         likelihood_user = compute_likelihood_userbased(ratings, u, i, y, alpha, R)
#         likelihood_item = compute_likelihood_itembased(ratings, u, i, y, alpha, R)

#         all_likelihood_user.append(likelihood_user)
#         all_likelihood_item.append(likelihood_item)

#         if mode == 'user':
#             score = prior_user * likelihood_user
#         elif mode == 'item':
#             score = prior_item * likelihood_item
#         else:  # hybrid
#             len_Iu = sum(1 for j in range(len(ratings[0])) if ratings[u][j] != 0 and j != i)
#             len_Ui = sum(1 for v in range(len(ratings)) if v != u and ratings[v][i] != 0)

#             score_user = (prior_user * likelihood_user) ** (1 / (1 + len_Iu)) if len_Iu > 0 else 0
#             score_item = (prior_item * likelihood_item) ** (1 / (1 + len_Ui)) if len_Ui > 0 else 0
#             score = score_user * score_item

#         scores.append(score)
#         all_combined.append(score)

#     predicted_rating = scores.index(max(scores)) + 1

#     return predicted_rating, {
#         'scores': scores,
#         'likelihood_user': all_likelihood_user,
#         'likelihood_item': all_likelihood_item,
#         'combined_score': all_combined
#     }



# import numpy as np
# import pandas as pd

# def compute_priors(ratings, alpha=0.01, R=5):
#     num_users = len(ratings)
#     num_items = len(ratings[0])
#     rating_values = list(range(1, R + 1))

#     prior_userbased = [[0 for _ in range(num_items)] for _ in rating_values]
#     prior_itembased = [[0 for _ in range(num_users)] for _ in rating_values]

#     for y in rating_values:
#         y_index = y - 1

#         # Prior user-based (per item j)
#         for j in range(num_items):
#             count_y = 0
#             count_nonzero = 0
#             for u in range(num_users):
#                 r = ratings[u][j]
#                 if r != 0:
#                     count_nonzero += 1
#                     if r == y:
#                         count_y += 1
#             prior_userbased[y_index][j] = (count_y + alpha) / (count_nonzero + alpha * R)

#         # Prior item-based (per user u)
#         for u in range(num_users):
#             count_y = 0
#             count_nonzero = 0
#             for j in range(num_items):
#                 r = ratings[u][j]
#                 if r != 0:
#                     count_nonzero += 1
#                     if r == y:
#                         count_y += 1
#             prior_itembased[y_index][u] = (count_y + alpha) / (count_nonzero + alpha * R)

#     return prior_userbased, prior_itembased


# def compute_likelihood_userbased(ratings, u, i, y, alpha=0.01, R=5):
#     num_users = len(ratings)
#     num_items = len(ratings[0])
#     Iu = [j for j in range(num_items) if j != i and ratings[u][j] != 0]
#     product = 1.0

#     for j in Iu:
#         k = ratings[u][j]
#         count_joint = 0
#         count_cond = 0
#         for v in range(num_users):
#             if ratings[v][i] == y:
#                 if ratings[v][j] != 0:
#                     count_cond += 1
#                     if ratings[v][j] == k:
#                         count_joint += 1
#         prob = (count_joint + alpha) / (count_cond + alpha * R)
#         # print(prob)
#         product *= prob
#     # print("======")
#     # print(product, end="\n\n")

#     return product


# def compute_likelihood_itembased(ratings, u, i, y, alpha=0.01, R=5):
#     num_users = len(ratings)
#     num_items = len(ratings[0])
#     Ui = [v for v in range(num_users) if v != u and ratings[v][i] != 0]
#     product = 1.0

#     for v in Ui:
#         k = ratings[v][i]
#         count_joint = 0
#         count_cond = 0
#         for j in range(num_items):
#             if ratings[u][j] == y:
#                 if ratings[v][j] != 0:
#                     count_cond += 1
#                     if ratings[v][j] == k:
#                         count_joint += 1
#         prob = (count_joint + alpha) / (count_cond + alpha * R)
#         product *= prob

#     return product


# def predict_rating(ratings, u, i, prior_userbased, prior_itembased, alpha=0.01, R=5, mode='hybrid'):
#     scores = []
#     all_likelihood_user = []  
#     all_likelihood_item = []
#     all_combined = []  

#     for y in range(1, R + 1):
#         y_index = y - 1
#         prior_user = prior_userbased[y_index][i]
#         prior_item = prior_itembased[y_index][u]

#         likelihood_user = compute_likelihood_userbased(ratings, u, i, y, alpha, R)
#         likelihood_item = compute_likelihood_itembased(ratings, u, i, y, alpha, R)

#         all_likelihood_user.append(likelihood_user)
#         all_likelihood_item.append(likelihood_item)

#         if mode == 'user':
#             score = prior_user * likelihood_user
#         elif mode == 'item':
#             score = prior_item * likelihood_item
#         else:  # hybrid
#             len_Iu = sum(1 for j in range(len(ratings[0])) if ratings[u][j] != 0 and j != i)
#             len_Ui = sum(1 for v in range(len(ratings)) if v != u and ratings[v][i] != 0)

#             score_user = (prior_user * likelihood_user) ** (1 / (1 + len_Iu)) if len_Iu > 0 else 0
#             score_item = (prior_item * likelihood_item) ** (1 / (1 + len_Ui)) if len_Ui > 0 else 0
#             score = score_user * score_item

#         scores.append(score)
#         all_combined.append(score)

#     predicted_rating = scores.index(max(scores)) + 1

#     return predicted_rating, {
#         'scores': scores,
#         'likelihood_user': all_likelihood_user,
#         'likelihood_item': all_likelihood_item,
#         'combined_score': all_combined
#     }




# def load_movielens_1m(path="./ml-1m/ratings.dat"):
#     df = pd.read_csv(path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])

#     num_users = df['user'].nunique()
#     num_items = df['item'].nunique()

#     user_map = {uid: idx for idx, uid in enumerate(df['user'].unique())}
#     item_map = {iid: idx for idx, iid in enumerate(df['item'].unique())}

#     ratings = np.zeros((num_users, num_items), dtype=int)
#     for _, row in df.iterrows():
#         u = user_map[row['user']]
#         i = item_map[row['item']]
#         ratings[u][i] = int(row['rating'])

#     return ratings, user_map, item_map


# def train_test_split_matrix(ratings, test_ratio=0.1, seed=42):
#     np.random.seed(seed)
#     train = ratings.copy()
#     test = []

#     for u in range(ratings.shape[0]):
#         items_rated = np.where(ratings[u] > 0)[0]
#         if len(items_rated) == 0:
#             continue
#         test_size = max(1, int(len(items_rated) * test_ratio))
#         test_items = np.random.choice(items_rated, size=test_size, replace=False)
#         for i in test_items:
#             test.append((u, i, ratings[u][i]))  # simpan ground truth
#             train[u][i] = 0  # kosongkan di train

#     return train, test


# ratings_full, user_map, item_map = load_movielens_1m("./ml-1m/ratings.dat")
# ratings_train, test_set = train_test_split_matrix(ratings_full, test_ratio=0.2)


# # Komputasi prior
# prior_userbased, prior_itembased = compute_priors(ratings_train)

# y_true = []
# y_pred = []

# for u, i, actual in tqdm(test_set):
#     pred, _ = predict_rating(ratings_train, u, i, prior_userbased, prior_itembased, mode='hybrid')
#     y_true.append(actual)
#     y_pred.append(pred)
    
    

# import math

# rmse = math.sqrt(mean_squared_error(y_true, y_pred))
# mae = mean_absolute_error(y_true, y_pred)

# print("RMSE:", rmse)
# print("MAE :", mae)




def compute_priors(ratings,plausible_rating, alpha=0.01, R=8):
    num_users = len(ratings)
    num_items = len(ratings[0])
    # rating_values = list(range(1, R + 1))
    rating_values = plausible_rating

    prior_userbased = [[0 for _ in range(num_items)] for _ in rating_values]
    prior_itembased = [[0 for _ in range(num_users)] for _ in rating_values]

    y_index = 0
    for y in rating_values:
        y_index = y_index

        # Prior user-based (per item j)
        for j in range(num_items):
            count_y = 0
            count_nonzero = 0
            for u in range(num_users):
                r = ratings[u][j]   
                if r != 0:
                    count_nonzero += 1
                    if r == y:
                        count_y += 1
            prior_userbased[y_index][j] = (count_y + alpha) / (count_nonzero + alpha * R)

        # Prior item-based (per user u)
        for u in range(num_users):
            count_y = 0
            count_nonzero = 0
            for j in range(num_items):
                r = ratings[u][j]
                if r != 0:
                    count_nonzero += 1
                    if r == y:
                        count_y += 1
            prior_itembased[y_index][u] = (count_y + alpha) / (count_nonzero + alpha * R)
        y_index = y_index + 1

    return prior_userbased, prior_itembased

def compute_likelihood_userbased(ratings, u, i, y, alpha=0.01, R=8):
    num_users = len(ratings)
    num_items = len(ratings[0])
    Iu = [j for j in range(num_items) if j != i and ratings[u][j] != 0]
    product = 1.0

    for j in Iu:
        k = ratings[u][j]
        count_joint = 0
        count_cond = 0
        for v in range(num_users):
            if ratings[v][i] == y:
                if ratings[v][j] != 0:
                    count_cond += 1
                    if ratings[v][j] == k:
                        count_joint += 1
        prob = (count_joint + alpha) / (count_cond + alpha * R)
        # print(prob)
        product *= prob
    # print("======")
    # print(product, end="\n\n")

    return product

def compute_likelihood_itembased(ratings, u, i, y, alpha=0.01, R=8):
    num_users = len(ratings)
    num_items = len(ratings[0])
    Ui = [v for v in range(num_users) if v != u and ratings[v][i] != 0]
    product = 1.0

    for v in Ui:
        k = ratings[v][i]
        count_joint = 0
        count_cond = 0
        for j in range(num_items):
            if ratings[u][j] == y:
                if ratings[v][j] != 0:
                    count_cond += 1
                    if ratings[v][j] == k:
                        count_joint += 1
        prob = (count_joint + alpha) / (count_cond + alpha * R)
        product *= prob

    return product


def predict_rating(ratings, u, i, prior_userbased, prior_itembased,plausible_rating, alpha=0.01, mode='hybrid'):
    scores = []
    all_likelihood_user = []  
    all_likelihood_item = []
    all_combined = []
    R = len(plausible_rating)  

    y_index = 0
    for y in plausible_rating:
        prior_user = prior_userbased[y_index][i]
        prior_item = prior_itembased[y_index][u]

        likelihood_user = compute_likelihood_userbased(ratings, u, i, y, alpha, R)
        likelihood_item = compute_likelihood_itembased(ratings, u, i, y, alpha, R)

        all_likelihood_user.append(likelihood_user)
        all_likelihood_item.append(likelihood_item)

        if mode == 'user':
            score = prior_user * likelihood_user
        elif mode == 'item':
            score = prior_item * likelihood_item
        else:  # hybrid
            len_Iu = sum(1 for j in range(len(ratings[0])) if ratings[u][j] != 0 and j != i)
            len_Ui = sum(1 for v in range(len(ratings)) if v != u and ratings[v][i] != 0)

            score_user = (prior_user * likelihood_user) ** (1 / (1 + len_Iu)) if len_Iu > 0 else 0
            score_item = (prior_item * likelihood_item) ** (1 / (1 + len_Ui)) if len_Ui > 0 else 0
            score = score_user * score_item

        scores.append(score)
        all_combined.append(score)
        y_index = y_index + 1

    predicted_rating = scores.index(max(scores)) + 1

    return predicted_rating, {
        'scores': scores,
        'likelihood_user': all_likelihood_user,
        'likelihood_item': all_likelihood_item,
        'combined_score': all_combined
    }

import numpy as np
import pandas as pd

def load_filmtrust(path):
    df = pd.read_csv(path, sep=' ', engine='python', names=['user', 'item', 'rating'])

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()
    # print(df['rating'].unique())    
    plausible_rating = df['rating'].unique()

    user_map = {uid: idx for idx, uid in enumerate(df['user'].unique())}
    item_map = {iid: idx for idx, iid in enumerate(df['item'].unique())}

    ratings = np.zeros((num_users, num_items), dtype=int)
    for _, row in df.iterrows():
        u = user_map[row['user']]
        i = item_map[row['item']]
        ratings[u][i] = int(row['rating'])

    return ratings, user_map, item_map, plausible_rating

def train_test_split_matrix(ratings, test_ratio=0.1, seed=42):
    np.random.seed(seed)
    train = ratings.copy()
    test = []

    for u in range(ratings.shape[0]):
        items_rated = np.where(ratings[u] > 0)[0]
        if len(items_rated) == 0:
            continue
        test_size = max(1, int(len(items_rated) * test_ratio))
        test_items = np.random.choice(items_rated, size=test_size, replace=False)
        for i in test_items:
            test.append((u, i, ratings[u][i]))  # simpan ground truth
            train[u][i] = 0  # kosongkan di train

    return train, test

ratings_full, user_map, item_map,pR = load_filmtrust("./film-trust/ratings.txt")

num_r = len(pR)
pR

pR.sort()
pR


ratings_train, test_set = train_test_split_matrix(ratings_full, test_ratio=0.2)


# Komputasi prior
prior_userbased, prior_itembased = compute_priors(ratings_train,pR)



# Prediksi dan evaluasi
from tqdm import tqdm

y_true = []
y_pred = []

for u, i, actual in tqdm(test_set):
    pred, _ = predict_rating(ratings_train, u, i, prior_userbased, prior_itembased,plausible_rating= pR, mode='hybrid')
    y_true.append(actual)
    y_pred.append(pred)