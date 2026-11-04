import pathlib


RAIZ = pathlib.Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_SAIDA = RAIZ / "output"
DIR_MODELOS = DIR_SAIDA / "models"
DIR_METRICAS = DIR_SAIDA / "metrics"
DIR_FIGURAS = DIR_SAIDA / "figures"
DIR_PREDICOES = DIR_SAIDA / "predictions"

ARQ_COM_LABEL = DIR_DADOS / "02_treino_sinais_vitais_com_label.csv"


COLUNAS_COM_LABEL = [
    "id",
    "pSist",
    "pDiast",
    "qPA",
    "pulso",
    "resp",
    "gravidade",
    "classe",
]

FEATURES = ["qPA", "pulso", "resp"]
ALVO_REGRESSAO = "gravidade"
ALVO_CLASSIFICACAO = "classe"

CLASSES = {
    1: "crítico",
    2: "instável",
    3: "potencialmente estável",
    4: "estável",
}
IDS_CLASSES = [1, 2, 3, 4]

CORTES_GRAVIDADE = [25, 50, 75]


SEED = 42
FRAC_TESTE = 0.2
N_FOLDS = 5

METRICA_CLASSIFICACAO = "f1_macro"
METRICA_REGRESSAO = "neg_root_mean_squared_error"
