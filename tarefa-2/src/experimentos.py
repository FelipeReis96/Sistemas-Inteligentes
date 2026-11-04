import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import models
from preprocess import prepara_dados


VARIACOES_MLP = {
    "hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)],
    "learning_rate_init": [1e-1, 1e-2, 1e-3, 1e-4],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "lbfgs", "sgd"],
}

VARIACOES_RF = {
    "n_estimators": [10, 50, 100, 300, 500],
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": [1, 2, 3],
}

EXPERIMENTOS = {
    "mlp": {
        "fabricas": {
            "classificacao": models.mlp_classificador,
            "regressao": models.mlp_regressor,
        },
        "normaliza": True,
        "base": {"max_iter": 1000},
        "variacoes": {
            "classificacao": VARIACOES_MLP,
            "regressao": VARIACOES_MLP,
        },
    },
    "rf": {
        "fabricas": {
            "classificacao": models.rf_classificador,
            "regressao": models.rf_regressor,
        },
        "normaliza": False,
        "base": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "max_features": "sqrt",
        },
        "variacoes": {
            "classificacao": {
                **VARIACOES_RF,
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"],
            },
            "regressao": VARIACOES_RF,
        },
    },
}

NOMES = {"mlp": "MLP", "rf": "RF"}
CAMPOS_CSV = ["modelo", "tarefa", "parametro", "valor", "metrica", "media", "desvio"]


def _monta(modelo, estimador):
    if EXPERIMENTOS[modelo]["normaliza"]:
        return Pipeline([("scaler", StandardScaler()), ("mlp", estimador)])
    return estimador


def roda_bateria(modelo, tarefa, dados, variacoes=None):
    exp = EXPERIMENTOS[modelo]
    fabrica = exp["fabricas"][tarefa]
    variacoes = variacoes or exp["variacoes"][tarefa]

    if tarefa == "classificacao":
        scoring = config.METRICA_CLASSIFICACAO
        y_treino = dados.y_classe_treino
        nome_metrica = "F1 macro"
        maior_melhor = True
    else:
        scoring = config.METRICA_REGRESSAO
        y_treino = dados.y_grav_treino
        nome_metrica = "RMSE"
        maior_melhor = False

    linhas = []
    for parametro, valores in variacoes.items():
        print(f"\n[{NOMES[modelo]} {tarefa}] variando {parametro} ...")
        medias, desvios, rotulos = [], [], []

        for valor in valores:
            estimador = fabrica(**{**exp["base"], parametro: valor})
            resultado = cross_validate(
                _monta(modelo, estimador),
                dados.X_treino, y_treino,
                scoring=scoring,
                cv=dados.folds,
                n_jobs=-1,
            )
            notas = resultado["test_score"]
            if not maior_melhor:
                notas = -notas

            media, desvio = float(np.mean(notas)), float(np.std(notas))
            medias.append(media)
            desvios.append(desvio)
            rotulos.append(str(valor))
            linhas.append({
                "modelo": modelo,
                "tarefa": tarefa,
                "parametro": parametro,
                "valor": str(valor),
                "metrica": nome_metrica,
                "media": round(media, 4),
                "desvio": round(desvio, 4),
            })
            print(f"    {valor}: {media:.4f} ± {desvio:.4f}")

        _plota_curva(modelo, tarefa, parametro, rotulos, medias, desvios,
                     nome_metrica, maior_melhor)

    return linhas


def _plota_curva(modelo, tarefa, parametro, rotulos, medias, desvios,
                 nome_metrica, maior_melhor):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(rotulos))
    ax.errorbar(x, medias, yerr=desvios, marker="o", capsize=4)

    idx = int(np.argmax(medias) if maior_melhor else np.argmin(medias))
    ax.scatter([idx], [medias[idx]], color="red", zorder=5,
               label=f"melhor: {rotulos[idx]}")

    ax.set_xticks(list(x), rotulos, rotation=30, ha="right")
    ax.set_xlabel(parametro)
    ax.set_ylabel(f"{nome_metrica} (validação cruzada)")
    ax.set_title(f"{NOMES[modelo]} {tarefa} — variando {parametro}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = config.DIR_FIGURAS / f"bateria_{modelo}_{tarefa}_{parametro}.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    figura: {caminho}")


def salva_csv(linhas, modelos_rodados):
    config.DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    caminho = config.DIR_METRICAS / "baterias.csv"

    antigas = []
    if caminho.exists():
        with open(caminho, encoding="utf-8", newline="") as f:
            for linha in csv.DictReader(f):
                linha.setdefault("modelo", "mlp")
                if linha["modelo"] not in modelos_rodados:
                    antigas.append({c: linha.get(c, "") for c in CAMPOS_CSV})

    with open(caminho, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        escritor.writerows(antigas + linhas)
    print(f"\nTabela salva em {caminho} ({len(antigas) + len(linhas)} linhas)")


def main():
    modelos_rodados = sys.argv[1:] or list(EXPERIMENTOS)
    for m in modelos_rodados:
        if m not in EXPERIMENTOS:
            sys.exit(f"Modelo '{m}' não existe. Opções: {', '.join(EXPERIMENTOS)}")

    dados = prepara_dados(verbose=True)
    linhas = []
    for modelo in modelos_rodados:
        linhas += roda_bateria(modelo, "classificacao", dados)
        linhas += roda_bateria(modelo, "regressao", dados)
    salva_csv(linhas, modelos_rodados)


if __name__ == "__main__":
    main()
