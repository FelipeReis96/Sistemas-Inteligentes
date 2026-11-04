import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import dataset

CORES_CLASSES = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4", 4: "#2ca02c"}

FAIXAS = {
    "qPA": [-9, -6, -3, -0.001, 0.001, 3, 6, 9],
    "pulso": [0, 40, 60, 80, 100, 120, 140, 160, 200],
    "resp": [0, 4, 8, 12, 16, 20, 22],
}

ROTULOS = {
    "qPA": "qPA",
    "pulso": "Pulso (bpm)",
    "resp": "Respiração (rpm)",
}


def mostra_estatisticas(df):
    dataset.resumo_dataset(df)

    print("\nEstatísticas dos atributos:")
    print(df[config.FEATURES + [config.ALVO_REGRESSAO]].describe().round(2).to_string())

    print("\nMédia de cada atributo por classe:")
    print(df.groupby("classe")[config.FEATURES + [config.ALVO_REGRESSAO]]
          .mean().round(1).to_string())

    print("\nCorrelação linear com a gravidade:")
    print(df[config.FEATURES].corrwith(df[config.ALVO_REGRESSAO]).round(2).to_string())

    print("\nGravidade média por faixa de cada atributo:")
    for atributo, faixas in FAIXAS.items():
        por_faixa = df.groupby(pd.cut(df[atributo], faixas), observed=True)[
            config.ALVO_REGRESSAO].mean().round(1)
        print(f"\n{atributo}:")
        print(por_faixa.to_string())

    print(f"\nVítimas com qPA exatamente 0: {(df['qPA'] == 0).sum()}")


def plota_sinais_vs_gravidade(df):
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

    for ax, atributo in zip(eixos, config.FEATURES):
        for classe in config.IDS_CLASSES:
            parte = df[df["classe"] == classe]
            ax.scatter(parte[atributo], parte[config.ALVO_REGRESSAO],
                       s=5, alpha=0.6, color=CORES_CLASSES[classe],
                       label=f"{classe} - {config.CLASSES[classe]} (n={len(parte)})")
        for corte in config.CORTES_GRAVIDADE:
            ax.axhline(corte, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel(ROTULOS[atributo])

    eixos[0].set_ylabel("Gravidade")
    handles, labels = eixos[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               markerscale=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    for extensao in ("png", "pdf"):
        caminho = config.DIR_FIGURAS / f"dados_sinais_vs_gravidade.{extensao}"
        fig.savefig(caminho, dpi=200)
        print(f"Figura salva em {caminho}")
    plt.close(fig)


def reconstitui_cortes():
    from sklearn.tree import DecisionTreeClassifier
    from preprocess import prepara_dados

    dados = prepara_dados()
    gravidade = dados.y_grav_treino.to_frame()
    classe = dados.y_classe_treino

    arvore = DecisionTreeClassifier(max_leaf_nodes=4, random_state=config.SEED)
    arvore.fit(gravidade, classe)

    nos = arvore.tree_
    limites = sorted(nos.threshold[nos.feature >= 0])

    print("\n" + "=" * 50)
    print("RECONSTITUIÇÃO DOS LIMITES ENTRE AS CLASSES")
    print("=" * 50)
    print("Limites encontrados pela árvore: "
          + ", ".join(f"{l:.2f}" for l in limites))
    print(f"Acerto no treino: {arvore.score(gravidade, classe):.0%}")
    print(f"Arredondados: {[round(l) for l in limites]} "
          f"(no config.py: {config.CORTES_GRAVIDADE})")


if __name__ == "__main__":
    df = dataset.carrega_com_label()
    mostra_estatisticas(df)
    plota_sinais_vs_gravidade(df)
    reconstitui_cortes()
