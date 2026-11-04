import pandas as pd
import config


def carrega_com_label(verbose=False):
    df = pd.read_csv(config.ARQ_COM_LABEL, header=None, names=config.COLUNAS_COM_LABEL)

    n_nulos = int(df.isnull().sum().sum())
    if n_nulos > 0:
        antes = len(df)
        df = df.dropna()
        print(f"Aviso: {antes - len(df)} linhas com valores nulos foram removidas")

    colunas = ["id", *config.FEATURES, config.ALVO_REGRESSAO, config.ALVO_CLASSIFICACAO]
    df = df[colunas].copy()
    df["classe"] = df["classe"].astype(int)
    df["id"] = df["id"].astype(int)

    if verbose:
        resumo_dataset(df)
    return df


def resumo_dataset(df):
    print("=" * 50)
    print("DATASET")
    print("=" * 50)
    print(f"{len(df)} vítimas")
    print(f"Valores nulos: {int(df.isnull().sum().sum())}")

    print("\nDistribuição das classes:")
    contagem = df[config.ALVO_CLASSIFICACAO].value_counts().sort_index()
    for classe, qtd in contagem.items():
        pct = 100 * qtd / len(df)
        print(f"  {classe} ({config.CLASSES[classe]}): {qtd} ({pct:.1f}%)")
