
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Pour LightGBM — à ajouter dans params_finaux après chargement
params_finaux["seed"]                 = SEED
params_finaux["random_state"]         = SEED
params_finaux["feature_fraction_seed"] = SEED
params_finaux["bagging_seed"]         = SEED
params_finaux["data_random_seed"]     = SEED
