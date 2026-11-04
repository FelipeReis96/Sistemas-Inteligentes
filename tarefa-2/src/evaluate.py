import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

import config
from preprocess import gravidade_para_classe


def avalia_classificacao(y_real, y_previsto, nome_modelo, verbose=True):
    y_real = np.asarray(y_real)
    y_previsto = np.asarray(y_previsto)

    precision, recall, f1, suporte = precision_recall_fscore_support(
        y_real, y_previsto, labels=config.IDS_CLASSES, zero_division=0
    )

    resultado = {
        "modelo": nome_modelo,
        "tipo": "classificacao",
        "acuracia": accuracy_score(y_real, y_previsto),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": f1_score(y_real, y_previsto, labels=config.IDS_CLASSES,
                             average="macro", zero_division=0),
        "por_classe": {
            str(c): {
                "precision": precision[k],
                "recall": recall[k],
                "f1": f1[k],
                "suporte": suporte[k],
            }
            for k, c in enumerate(config.IDS_CLASSES)
        },
        "matriz_confusao": confusion_matrix(
            y_real, y_previsto, labels=config.IDS_CLASSES
        ).tolist(),
    }

    if verbose:
        print(f"\n===== {nome_modelo} =====")
        print(f"Acurácia: {resultado['acuracia']:.4f}")
        print(f"F1 macro: {resultado['f1_macro']:.4f}\n")
        print(classification_report(
            y_real, y_previsto, labels=config.IDS_CLASSES,
            target_names=[f"{c} - {config.CLASSES[c]}" for c in config.IDS_CLASSES],
            digits=4, zero_division=0,
        ))

    return resultado


def plota_matriz_confusao(y_real, y_previsto, titulo, nome_arquivo):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_real, y_previsto, labels=config.IDS_CLASSES,
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    _salva_figura(fig, nome_arquivo)


def avalia_regressao(y_grav_real, y_grav_previsto, nome_modelo,
                     y_classe_real=None, verbose=True):
    y_grav_real = np.asarray(y_grav_real)
    y_grav_previsto = np.asarray(y_grav_previsto)

    resultado = {
        "modelo": nome_modelo,
        "tipo": "regressao",
        "rmse": float(np.sqrt(mean_squared_error(y_grav_real, y_grav_previsto))),
        "mae": mean_absolute_error(y_grav_real, y_grav_previsto),
        "r2": r2_score(y_grav_real, y_grav_previsto),
    }

    if verbose:
        print(f"\n===== {nome_modelo} =====")
        print(f"RMSE: {resultado['rmse']:.4f}")
        print(f"MAE:  {resultado['mae']:.4f}")
        print(f"R²:   {resultado['r2']:.4f}")

    if y_classe_real is not None:
        classe_prevista = gravidade_para_classe(y_grav_previsto)
        resultado["classificacao_via_gravidade"] = avalia_classificacao(
            y_classe_real, classe_prevista,
            f"{nome_modelo} (classe via cortes)", verbose=verbose,
        )

    return resultado


def plota_real_vs_previsto(y_grav_real, y_grav_previsto, titulo, nome_arquivo):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_grav_real, y_grav_previsto, s=12, alpha=0.6)

    limites = [0, 100]
    ax.plot(limites, limites, "k--", linewidth=1, label="previsão perfeita")
    for corte in config.CORTES_GRAVIDADE:
        ax.axvline(corte, color="gray", linewidth=0.6, alpha=0.5)
        ax.axhline(corte, color="gray", linewidth=0.6, alpha=0.5)

    ax.set_xlim(limites)
    ax.set_ylim(limites)
    ax.set_xlabel("Gravidade real")
    ax.set_ylabel("Gravidade prevista")
    ax.set_title(titulo)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _salva_figura(fig, nome_arquivo)


def importancia_atributos(modelo, X_teste, y_teste, nome_modelo, nome_arquivo):
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        modelo, X_teste, y_teste,
        n_repeats=10,
        random_state=config.SEED,
    )
    importancias = dict(zip(X_teste.columns, r.importances_mean))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(list(importancias), list(importancias.values()),
            xerr=r.importances_std, color="steelblue")
    ax.set_xlabel("Queda no desempenho ao embaralhar")
    ax.set_title(f"Importância dos atributos - {nome_modelo}")
    fig.tight_layout()
    _salva_figura(fig, nome_arquivo)

    return {k: float(v) for k, v in importancias.items()}


def resumo_validacao_cruzada(grid, verbose=True):
    melhor = grid.best_index_
    media = grid.cv_results_["mean_test_score"][melhor]
    desvio = grid.cv_results_["std_test_score"][melhor]

    if media < 0:
        media = -media

    resumo = {
        "melhores_hiperparametros": grid.best_params_,
        "metrica_cv": grid.scoring,
        "cv_media": float(media),
        "cv_desvio": float(desvio),
    }

    if verbose:
        print("Melhores hiperparâmetros:")
        for nome, valor in grid.best_params_.items():
            print(f"  {nome}: {valor}")
        print(f"Validação cruzada ({grid.scoring}): {media:.4f} ± {desvio:.4f}")

    return resumo


def salva_resultados(resultado, nome_arquivo, resumo_cv=None):
    if resumo_cv is not None:
        resultado = {**resultado, "validacao_cruzada": resumo_cv}

    config.DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    caminho = config.DIR_METRICAS / f"{nome_arquivo}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False, default=_para_json)
    print(f"Resultados salvos em {caminho}")


def salva_previsoes(ids, y_real, y_previsto, nome_arquivo, tipo):
    config.DIR_PREDICOES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": ids, "real": np.asarray(y_real),
                       "previsto": np.asarray(y_previsto)})
    if tipo == "classificacao":
        df["acertou"] = df["real"] == df["previsto"]
    else:
        df["erro"] = df["previsto"] - df["real"]
        df["classe_real"] = gravidade_para_classe(df["real"])
        df["classe_prevista"] = gravidade_para_classe(df["previsto"])
    caminho = config.DIR_PREDICOES / f"{nome_arquivo}.csv"
    df.to_csv(caminho, index=False)


def tabela_comparativa(salvar=True):
    linhas = []
    for caminho in sorted(config.DIR_METRICAS.glob("*.json")):
        with open(caminho, encoding="utf-8") as f:
            r = json.load(f)

        if r["tipo"] == "classificacao":
            linhas.append(_linha_classificacao(r))
        else:
            linha = {"modelo": r["modelo"], "rmse": r["rmse"],
                     "mae": r["mae"], "r2": r["r2"]}
            if "classificacao_via_gravidade" in r:
                linha.update(_linha_classificacao(r["classificacao_via_gravidade"],
                                                  incluir_nome=False))
            linhas.append(linha)

    tabela = pd.DataFrame(linhas).set_index("modelo").round(4)
    if salvar:
        tabela.to_csv(config.DIR_METRICAS / "tabela_comparativa.csv")
    return tabela


def _linha_classificacao(r, incluir_nome=True):
    linha = {"modelo": r["modelo"]} if incluir_nome else {}
    linha.update({
        "acuracia": r["acuracia"],
        "precision_macro": r["precision_macro"],
        "recall_macro": r["recall_macro"],
        "f1_macro": r["f1_macro"],
        "recall_critico": r["por_classe"]["1"]["recall"],
        "recall_estavel": r["por_classe"]["4"]["recall"],
    })
    return linha


def _salva_figura(fig, nome_arquivo):
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = config.DIR_FIGURAS / f"{nome_arquivo}.png"
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    print(f"Figura salva em {caminho}")


def _para_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Não sei converter {type(obj)} pra JSON")
