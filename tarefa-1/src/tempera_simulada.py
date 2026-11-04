"""Têmpera Simulada aplicada às rotas do Caixeiro Viajante."""

import math
import random

from problema import calcular_custo_rota, validar_rota


def gerar_vizinha_por_troca(rota, gerador):
    """Cria uma rota vizinha trocando duas cidades de posição."""
    nova_rota = rota.copy()
    posicao_a, posicao_b = gerador.sample(range(len(rota)), 2)
    nova_rota[posicao_a], nova_rota[posicao_b] = (
        nova_rota[posicao_b],
        nova_rota[posicao_a],
    )
    return nova_rota


def gerar_vizinha_por_inversao(rota, gerador):
    """Cria uma vizinha invertendo um intervalo da rota."""
    while True:
        inicio, fim = sorted(gerador.sample(range(len(rota)), 2))
        if inicio != 0 or fim != len(rota) - 1:
            break

    nova_rota = rota.copy()
    nova_rota[inicio : fim + 1] = reversed(nova_rota[inicio : fim + 1])
    return nova_rota


def gerar_vizinha(rota, gerador, vizinhanca):
    """Aplica o operador de vizinhança escolhido para a execução."""
    if vizinhanca == "troca":
        return gerar_vizinha_por_troca(rota, gerador)
    if vizinhanca == "inversao":
        return gerar_vizinha_por_inversao(rota, gerador)
    raise ValueError("A vizinhança informada não existe.")


def deve_aceitar(diferenca, temperatura, gerador):
    """Aceita melhorias sempre e pioras conforme a regra de Metropolis."""
    if diferenca <= 0:
        # Se a vizinha for melhor ou igual a atual, aceitamos ela
        return True
    if temperatura <= 0:
        # Após muitas multiplicações, um float pode chegar exatamente a zero.
        return False

    # Quanto maior a piora em custo (diferenca), menor a probabilidade
    # Quanto maior a temperatura, que é decrescente, maior a probabilidade
    probabilidade = math.exp(-diferenca / temperatura)
    return gerador.random() < probabilidade


def executar_tempera_simulada(
    rota_inicial,
    distancias,
    temperatura_inicial,
    taxa_resfriamento,
    iteracoes,
    semente,
    vizinhanca="troca",
):
    """Executa a busca e retorna as rotas e medidas importantes da execução."""
    
    validar_rota(rota_inicial, len(distancias))

    gerador = random.Random(semente)
    rota_atual = rota_inicial.copy()
    custo_atual = calcular_custo_rota(rota_atual, distancias)
    melhor_rota = rota_atual.copy()
    melhor_custo = custo_atual
    temperatura = temperatura_inicial

    historico_melhor_custo = [melhor_custo]
    quantidade_aceitas = 0
    quantidade_pioras_aceitas = 0

    for _ in range(iteracoes):
        rota_vizinha = gerar_vizinha(rota_atual, gerador, vizinhanca)
        custo_vizinho = calcular_custo_rota(rota_vizinha, distancias)
        diferenca = custo_vizinho - custo_atual

        if deve_aceitar(diferenca, temperatura, gerador):
            rota_atual = rota_vizinha
            custo_atual = custo_vizinho
            quantidade_aceitas += 1
            if diferenca > 0:
                quantidade_pioras_aceitas += 1

        if custo_atual < melhor_custo:
            melhor_rota = rota_atual.copy()
            melhor_custo = custo_atual

        historico_melhor_custo.append(melhor_custo)
        temperatura *= taxa_resfriamento

    return {
        "melhor_rota": melhor_rota,
        "melhor_custo": melhor_custo,
        "rota_final": rota_atual,
        "custo_final": custo_atual,
        "historico_melhor_custo": historico_melhor_custo,
        "quantidade_avaliacoes": iteracoes + 1,
        "quantidade_aceitas": quantidade_aceitas,
        "quantidade_pioras_aceitas": quantidade_pioras_aceitas,
        "temperatura_final": temperatura,
    }
