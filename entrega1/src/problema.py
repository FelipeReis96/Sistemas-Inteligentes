"""Instâncias euclidianas, permutações e custo de ciclos fechados."""

import math
import random


def validar_quantidade_cidades(quantidade):
    """O recorte do trabalho exige pelo menos três cidades."""
    if type(quantidade) is not int or quantidade < 3:
        raise ValueError("O número de cidades deve ser um inteiro maior ou igual a três.")


def gerar_cidades(quantidade, semente, lado=100.0):
    """Retorna uma lista de cidades, cada uma representada por [x, y]."""
    validar_quantidade_cidades(quantidade)
    if not math.isfinite(lado) or lado <= 0:
        raise ValueError("O lado do quadrado deve ser positivo e finito.")

    # Um gerador separado permite repetir os sorteios sem afetar outros códigos.
    gerador = random.Random(semente)
    cidades = []
    for i in range(quantidade):
        x = gerador.uniform(0, lado)
        y = gerador.uniform(0, lado)
        cidades.append([x, y])
    return cidades


def calcular_distancias(cidades):
    """Monta uma matriz em que distancias[i][j] liga as cidades i e j."""
    validar_quantidade_cidades(len(cidades))
    for cidade in cidades:
        if len(cidade) != 2:
            raise ValueError("Cada cidade deve ter duas coordenadas: [x, y].")
        for coordenada in cidade:
            if not math.isfinite(coordenada):
                raise ValueError("As coordenadas devem ser finitas.")

    distancias = []
    for origem in cidades:
        linha = []
        for destino in cidades:
            # math.dist calcula a distância euclidiana entre os dois pontos.
            distancia = math.dist(origem, destino)
            if not math.isfinite(distancia):
                raise ValueError("As distâncias devem ser finitas.")
            linha.append(distancia)
        distancias.append(linha)
    return distancias


def validar_rota(rota, quantidade_cidades):
    """Verifica se a rota visita todos os índices de cidades uma única vez."""
    validar_quantidade_cidades(quantidade_cidades)
    if len(rota) != quantidade_cidades:
        raise ValueError("A rota deve ter um índice para cada cidade.")

    visitadas = [False] * quantidade_cidades
    for cidade in rota:
        if type(cidade) is not int:
            raise ValueError("Os índices das cidades devem ser inteiros.")
        if cidade < 0 or cidade >= quantidade_cidades:
            raise ValueError("A rota contém um índice de cidade inexistente.")
        if visitadas[cidade]:
            raise ValueError("A rota não pode repetir cidades.")
        visitadas[cidade] = True


def calcular_custo_rota(rota, distancias):
    """Soma o percurso usando a matriz retornada por calcular_distancias."""
    validar_rota(rota, len(distancias))
    custo = 0.0
    for i in range(len(rota) - 1):
        origem = rota[i]
        destino = rota[i + 1]
        custo += distancias[origem][destino]

    # Fecha o ciclo: da última cidade de volta à primeira.
    ultima = rota[-1]
    primeira = rota[0]
    custo += distancias[ultima][primeira]
    return custo


def gerar_rota_aleatoria(cidades, semente):
    """Embaralha os índices das cidades para criar uma rota inicial."""
    validar_quantidade_cidades(len(cidades))
    rota = list(range(len(cidades)))
    gerador = random.Random(semente)
    gerador.shuffle(rota)
    return rota


def gerar_rota_vizinho_mais_proximo(distancias, cidade_inicial=0):
    """Constrói uma rota escolhendo sempre a cidade não visitada mais próxima."""
    quantidade_cidades = len(distancias)
    validar_quantidade_cidades(quantidade_cidades)
    if type(cidade_inicial) is not int:
        raise ValueError("A cidade inicial deve ser representada por um inteiro.")
    if cidade_inicial < 0 or cidade_inicial >= quantidade_cidades:
        raise ValueError("A cidade inicial deve pertencer à instância.")

    rota = [cidade_inicial]
    nao_visitadas = list(range(quantidade_cidades))
    nao_visitadas.remove(cidade_inicial)

    while nao_visitadas:
        cidade_atual = rota[-1]
        proxima = min(
            nao_visitadas,
            key=lambda cidade: (distancias[cidade_atual][cidade], cidade),
        )
        rota.append(proxima)
        nao_visitadas.remove(proxima)

    return rota
