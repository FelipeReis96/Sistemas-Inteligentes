"""Demonstração da Têmpera Simulada e do AG no Caixeiro Viajante."""

import argparse

from algoritmo_genetico import executar_algoritmo_genetico, gerar_populacao
from problema import calcular_custo_rota, calcular_distancias, gerar_cidades, gerar_rota_aleatoria
from tempera_simulada import executar_tempera_simulada


QUANTIDADE_CIDADES = 10

SEMENTE_INSTANCIA = 42
SEMENTE_ROTA = 7
SEMENTE_BUSCA = 13

TEMPERATURA_INICIAL = 100.0
TAXA_RESFRIAMENTO = 0.995
ITERACOES = 1000

TAMANHO_POPULACAO = 50
SEMENTE_POPULACAO = 7
TAXA_CRUZAMENTO = 0.8
TAXA_MUTACAO = 0.05
GERACOES = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=["tempera", "ag", "ambos"],
        default="ambos",
        help="Algoritmo a executar (padrão: tempera).",
    )

    parser.add_argument(
        "--cities",
        type=int,
        default=QUANTIDADE_CIDADES,
        help=f"Número de cidades (mínimo 3; padrão: {QUANTIDADE_CIDADES}).",
    )
    parser.add_argument("--instance-seed", type=int, default=SEMENTE_INSTANCIA)
    parser.add_argument("--route-seed", type=int, default=SEMENTE_ROTA)
    parser.add_argument("--search-seed", type=int, default=SEMENTE_BUSCA)
    parser.add_argument("--temperature", type=float, default=TEMPERATURA_INICIAL)
    parser.add_argument("--cooling-rate", type=float, default=TAXA_RESFRIAMENTO)
    parser.add_argument("--iterations", type=int, default=ITERACOES)
    parser.add_argument("--population-size", type=int, default=TAMANHO_POPULACAO)
    parser.add_argument("--population-seed", type=int, default=SEMENTE_POPULACAO)
    parser.add_argument("--crossover-rate", type=float, default=TAXA_CRUZAMENTO)
    parser.add_argument("--mutation-rate", type=float, default=TAXA_MUTACAO)
    parser.add_argument("--generations", type=int, default=GERACOES)

    args = parser.parse_args()

    try:
        cidades = gerar_cidades(args.cities, semente=args.instance_seed)
        distancias = calcular_distancias(cidades)
        resultados = []

        if args.algorithm in ["tempera", "ambos"]:
            rota = gerar_rota_aleatoria(cidades, semente=args.route_seed)
            custo_inicial = calcular_custo_rota(rota, distancias)
            resultado = executar_tempera_simulada(
                rota,
                distancias,
                temperatura_inicial=args.temperature,
                taxa_resfriamento=args.cooling_rate,
                iteracoes=args.iterations,
                semente=args.search_seed,
            )
            resultados.append(("Têmpera Simulada", custo_inicial, resultado))

        if args.algorithm in ["ag", "ambos"]:
            populacao = gerar_populacao(
                cidades, args.population_size, semente=args.population_seed
            )
            resultado = executar_algoritmo_genetico(
                populacao,
                distancias,
                taxa_cruzamento=args.crossover_rate,
                taxa_mutacao=args.mutation_rate,
                geracoes=args.generations,
                semente=args.search_seed,
            )
            resultados.append(
                ("Algoritmo Genético", resultado["melhor_custo_inicial"], resultado)
            )
    except ValueError as error:
        parser.error(str(error))

    print(f"Instância: {len(cidades)} cidades | semente {args.instance_seed}")
    for nome, custo_inicial, resultado in resultados:
        melhor_custo = resultado["melhor_custo"]
        reducao = custo_inicial - melhor_custo
        percentual = 0.0
        if custo_inicial != 0:
            percentual = 100 * reducao / custo_inicial

        print(f"\n{nome}")
        print(f"  Custo inicial: {custo_inicial:.6f}")
        print(f"  Melhor custo:  {melhor_custo:.6f}")
        print(f"  Redução:        {reducao:.6f} ({percentual:.2f}%)")
        print(f"  Avaliações:     {resultado['quantidade_avaliacoes']}")


if __name__ == "__main__":
    main()
