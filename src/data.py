import numpy as np

# Cat/Dog Classifier

# Features per row (n=3):
# height_cm
# weight_kg
# ear_len_cm

# 0 = cat, 1 = dog

def load_data(m = 6, n = 3, seed=3):
    rng = np.random.default_rng(seed) # random gen

    m0 = m // 2 # cats
    m1 = m - m0 # dogs

    # cats: 
    height_cat = rng.uniform(20, 30, size=(m0, 1))
    weight_cat = rng.uniform(3, 6, size=(m0, 1))
    ear_len_cat = rng.uniform(3, 7, size=(m0, 1))
    X_cat = np.hstack([height_cat, weight_cat, ear_len_cat])
    
    height_dog = rng.uniform(30, 60, size=(m1, 1))
    weight_dog = rng.uniform(7, 30, size=(m1, 1))
    ear_len_dog = rng.uniform(6, 14, size=(m1, 1))
    X_dog = np.hstack([height_dog, weight_dog, ear_len_dog])


    X_train = np.vstack([X_cat, X_dog])
    y_train = np.concatenate([np.zeros(m0, dtype=int), np.ones(m1, dtype=int)])
    # print(Y_train)

    idx = rng.permutation(m)
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    return(X_train, y_train)


load_data()