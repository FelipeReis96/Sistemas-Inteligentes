import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({"font.size": 9})

LARGURA = 6.0

TECNICAS = [("rf", "RF"), ("mlp", "MLP")]


def _carrega_json(nome):
    with open(config.DIR_METRICAS / f"{nome}.json", encoding="utf-8") as f:
        return json.load(f)


def _salva(fig, nome):
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    for extensao in ("pdf", "png"):
        caminho = config.DIR_FIGURAS / f"{nome}.{extensao}"
        fig.savefig(caminho, dpi=200, bbox_inches="tight")
        print(f"Figura salva em {caminho}")
    plt.close(fig)


def _desenha_matriz(ax, matriz, titulo):
    matriz = np.array(matriz)
    proporcao = matriz / matriz.sum(axis=1, keepdims=True)

    ax.imshow(proporcao, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            cor = "white" if proporcao[i, j] > 0.6 else "black"
            ax.text(j, i, matriz[i, j], ha="center", va="center", color=cor)

    ax.set_xticks(range(4), config.IDS_CLASSES)
    ax.set_yticks(range(4), config.IDS_CLASSES)
    ax.set_title(titulo, fontsize=9)


def figura_matrizes():
    fig, eixos = plt.subplots(2, 2, figsize=(LARGURA * 0.75, LARGURA * 0.75))
    letras = iter("abcd")

    for linha, (chave, nome) in enumerate(TECNICAS):
        direta = _carrega_json(f"{chave}_classificador")
        via_grav = _carrega_json(f"{chave}_regressor")["classificacao_via_gravidade"]

        _desenha_matriz(eixos[linha, 0], direta["matriz_confusao"],
                        f"({next(letras)}) {nome}, classificação direta")
        _desenha_matriz(eixos[linha, 1], via_grav["matriz_confusao"],
                        f"({next(letras)}) {nome}, via gravidade")

    for ax in eixos[1, :]:
        ax.set_xlabel("Classe prevista")
    for ax in eixos[:, 0]:
        ax.set_ylabel("Classe real")

    fig.tight_layout()
    _salva(fig, "resultados_matrizes")


def figura_dispersao():
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, LARGURA * 0.45),
                              sharex=True, sharey=True)
    limites = [0, 100]

    for ax, (chave, nome), letra in zip(eixos, TECNICAS, "ab"):
        prev = pd.read_csv(config.DIR_PREDICOES / f"{chave}_regressor.csv")
        rmse = _carrega_json(f"{chave}_regressor")["rmse"]
        errou = prev["classe_real"] != prev["classe_prevista"]

        ax.scatter(prev.loc[~errou, "real"], prev.loc[~errou, "previsto"],
                   s=8, alpha=0.5, color="tab:blue", label="classe correta")
        ax.scatter(prev.loc[errou, "real"], prev.loc[errou, "previsto"],
                   s=18, marker="x", color="tab:red",
                   label=f"classe errada ({errou.sum()})")

        ax.plot(limites, limites, "k--", linewidth=0.8)
        for corte in config.CORTES_GRAVIDADE:
            ax.axvline(corte, color="gray", linewidth=0.5, alpha=0.6)
            ax.axhline(corte, color="gray", linewidth=0.5, alpha=0.6)

        ax.set_xlim(limites)
        ax.set_ylim(limites)
        ax.set_aspect("equal")
        ax.set_xlabel("Gravidade real")
        ax.set_title(f"({letra}) {nome} regressor, RMSE = {rmse:.2f}".replace(".", ","),
                     fontsize=9)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    eixos[0].set_ylabel("Gravidade prevista")
    fig.tight_layout()
    _salva(fig, "resultados_dispersao")


def figura_importancia():
    modelos = [
        ("rf_classificador", "RF classificador"),
        ("rf_regressor", "RF regressor"),
        ("mlp_classificador", "MLP classificador"),
        ("mlp_regressor", "MLP regressor"),
    ]
    rotulos_atributos = {"qPA": "qPA", "pulso": "Pulso", "resp": "Respiração"}

    fig, ax = plt.subplots(figsize=(LARGURA * 0.8, LARGURA * 0.35))
    largura_barra = 0.2
    x = np.arange(len(config.FEATURES))

    for k, (chave, nome) in enumerate(modelos):
        imp = _carrega_json(chave)["importancia_atributos"]
        total = sum(imp.values())
        relativa = [100 * imp[a] / total for a in config.FEATURES]
        ax.bar(x + (k - 1.5) * largura_barra, relativa, largura_barra, label=nome)

    ax.set_xticks(x, [rotulos_atributos[a] for a in config.FEATURES])
    ax.set_ylabel("Importância relativa (%)")
    ax.axhline(100 / 3, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    ax.set_ylim(0, 50)
    fig.tight_layout()
    _salva(fig, "resultados_importancia")


if __name__ == "__main__":
    figura_matrizes()
    figura_dispersao()
    figura_importancia()
