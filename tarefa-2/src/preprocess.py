from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
import dataset


def separa_entradas_e_alvos(df):
    df = df.set_index("id")
    X = df[config.FEATURES]
    y_classe = df[config.ALVO_CLASSIFICACAO]
    y_grav = df[config.ALVO_REGRESSAO]
    return X, y_classe, y_grav


def divide_treino_teste(X, y_classe, y_grav):
    return train_test_split(
        X, y_classe, y_grav,
        test_size=config.FRAC_TESTE,
        stratify=y_classe,
        random_state=config.SEED,
    )


def gera_folds(X_treino, y_classe_treino):
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    return list(skf.split(X_treino, y_classe_treino))


def gravidade_para_classe(gravidade):
    return np.digitize(np.asarray(gravidade), config.CORTES_GRAVIDADE) + 1


@dataclass
class DadosPreparados:
    X_treino: pd.DataFrame
    X_teste: pd.DataFrame
    y_classe_treino: pd.Series
    y_classe_teste: pd.Series
    y_grav_treino: pd.Series
    y_grav_teste: pd.Series
    folds: list


def prepara_dados(verbose=False):
    df = dataset.carrega_com_label(verbose=verbose)
    X, y_classe, y_grav = separa_entradas_e_alvos(df)

    (X_treino, X_teste,
     y_classe_treino, y_classe_teste,
     y_grav_treino, y_grav_teste) = divide_treino_teste(X, y_classe, y_grav)

    folds = gera_folds(X_treino, y_classe_treino)

    if verbose:
        print("\n" + "=" * 50)
        print("DIVISÃO TREINO / TESTE")
        print("=" * 50)
        print(f"Treino: {len(X_treino)} vítimas | Teste: {len(X_teste)} vítimas")
        print(f"Validação cruzada: {config.N_FOLDS} folds de ~{len(X_treino) // config.N_FOLDS} vítimas")
        print("\nProporção das classes (treino x teste):")
        prop_treino = y_classe_treino.value_counts(normalize=True).sort_index()
        prop_teste = y_classe_teste.value_counts(normalize=True).sort_index()
        for classe in prop_treino.index:
            print(f"  classe {classe}: {100 * prop_treino[classe]:.1f}% x "
                  f"{100 * prop_teste[classe]:.1f}%")

    return DadosPreparados(
        X_treino, X_teste,
        y_classe_treino, y_classe_teste,
        y_grav_treino, y_grav_teste,
        folds,
    )


if __name__ == "__main__":
    prepara_dados(verbose=True)
