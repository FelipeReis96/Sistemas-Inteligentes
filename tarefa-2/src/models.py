from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def rf_classificador(**kwargs):
    params = dict(
        random_state=config.SEED,
        n_jobs=-1,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def rf_regressor(**kwargs):
    params = dict(
        criterion="squared_error",
        random_state=config.SEED,
        n_jobs=-1,
    )
    params.update(kwargs)
    return RandomForestRegressor(**params)


GRADE_RF_CLASSIFICADOR = {
    "n_estimators": [100, 300],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", None],
    "class_weight": [None, "balanced"],
}

GRADE_RF_REGRESSOR = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", None],
}


def mlp_classificador(**kwargs):
    params = dict(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        learning_rate_init=1e-3,
        random_state=config.SEED,
    )
    params.update(kwargs)
    return MLPClassifier(**params)


def mlp_regressor(**kwargs):
    params = dict(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        learning_rate_init=1e-3,
        random_state=config.SEED,
    )
    params.update(kwargs)
    return MLPRegressor(**params)


def mlp_classificador_normalizado(**kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp_classificador(**kwargs)),
    ])


def mlp_regressor_normalizado(**kwargs):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp_regressor(**kwargs)),
    ])


GRADE_MLP = [
    {
        "mlp__solver": ["adam"],
        "mlp__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
        "mlp__learning_rate_init": [1e-1, 1e-2, 1e-3],
        "mlp__alpha": [1e-4, 1e-2],
        "mlp__activation": ["relu", "tanh"],
    },
    {
        "mlp__solver": ["lbfgs"],
        "mlp__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
        "mlp__alpha": [1e-4, 1e-2],
        "mlp__activation": ["relu", "tanh"],
    },
]

GRADE_MLP_CLASSIFICADOR = GRADE_MLP
GRADE_MLP_REGRESSOR = GRADE_MLP


MODELOS = {
    "rf_classificador": {
        "nome": "RF classificador",
        "tipo": "classificacao",
        "cria": rf_classificador,
        "grade": GRADE_RF_CLASSIFICADOR,
    },
    "rf_regressor": {
        "nome": "RF regressor",
        "tipo": "regressao",
        "cria": rf_regressor,
        "grade": GRADE_RF_REGRESSOR,
    },
    "mlp_classificador": {
        "nome": "MLP classificador",
        "tipo": "classificacao",
        "cria": mlp_classificador_normalizado,
        "grade": GRADE_MLP_CLASSIFICADOR,
    },
    "mlp_regressor": {
        "nome": "MLP regressor",
        "tipo": "regressao",
        "cria": mlp_regressor_normalizado,
        "grade": GRADE_MLP_REGRESSOR,
    },
}
