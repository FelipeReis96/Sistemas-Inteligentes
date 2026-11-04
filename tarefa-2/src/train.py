import sys
import time

import joblib
from sklearn.model_selection import GridSearchCV, ParameterGrid

import config
import evaluate
from models import MODELOS
from preprocess import gravidade_para_classe, prepara_dados


def treina_e_avalia(chave, dados):
    spec = MODELOS[chave]
    nome, tipo = spec["nome"], spec["tipo"]

    print("\n" + "#" * 60)
    print(f"# {nome}")
    print("#" * 60)

    if tipo == "classificacao":
        y_treino, y_teste = dados.y_classe_treino, dados.y_classe_teste
        metrica = config.METRICA_CLASSIFICACAO
    else:
        y_treino, y_teste = dados.y_grav_treino, dados.y_grav_teste
        metrica = config.METRICA_REGRESSAO

    n_config = len(ParameterGrid(spec["grade"]))
    print(f"Testando {n_config} configurações x {config.N_FOLDS} folds "
          f"= {n_config * config.N_FOLDS} treinos")

    grid = GridSearchCV(
        spec["cria"](),
        spec["grade"],
        cv=dados.folds,
        scoring=metrica,
        refit=True,
        n_jobs=-1,
    )
    inicio = time.time()
    grid.fit(dados.X_treino, y_treino)
    print(f"GridSearch levou {time.time() - inicio:.1f} s")

    resumo_cv = evaluate.resumo_validacao_cruzada(grid)
    modelo = grid.best_estimator_

    y_previsto = modelo.predict(dados.X_teste)

    if tipo == "classificacao":
        resultado = evaluate.avalia_classificacao(y_teste, y_previsto, nome)
        evaluate.plota_matriz_confusao(y_teste, y_previsto, nome, f"{chave}_matriz")
    else:
        resultado = evaluate.avalia_regressao(
            y_teste, y_previsto, nome, y_classe_real=dados.y_classe_teste
        )
        evaluate.plota_real_vs_previsto(y_teste, y_previsto, nome, f"{chave}_dispersao")
        evaluate.plota_matriz_confusao(
            dados.y_classe_teste,
            gravidade_para_classe(y_previsto),
            f"{nome} (classe via cortes)",
            f"{chave}_matriz",
        )

    resultado["importancia_atributos"] = evaluate.importancia_atributos(
        modelo, dados.X_teste, y_teste, nome, f"{chave}_importancia"
    )

    evaluate.salva_previsoes(dados.X_teste.index, y_teste, y_previsto, chave, tipo)
    evaluate.salva_resultados(resultado, chave, resumo_cv)

    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, config.DIR_MODELOS / f"{chave}.joblib")


def main():
    chaves = sys.argv[1:] or list(MODELOS)
    for chave in chaves:
        if chave not in MODELOS:
            sys.exit(f"Modelo '{chave}' não existe. Opções: {', '.join(MODELOS)}")

    dados = prepara_dados(verbose=True)

    for chave in chaves:
        treina_e_avalia(chave, dados)

    print("\n" + "=" * 60)
    print("TABELA COMPARATIVA (tudo que está em output/metrics)")
    print("=" * 60)
    print(evaluate.tabela_comparativa().to_string())


if __name__ == "__main__":
    main()
