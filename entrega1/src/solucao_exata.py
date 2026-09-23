"""Solução exata do Caixeiro Viajante pelo algoritmo de Held-Karp."""

import itertools


def criar_mascara(cidades):
    """Representa um conjunto de cidades, exceto a cidade 0, como bits."""
    mascara = 0
    for cidade in cidades:
        mascara |= 1 << (cidade - 1)
    return mascara


def resolver_exato(distancias):
    """Retorna uma rota ótima e seu custo pelo algoritmo de Held-Karp.

    A cidade 0 é fixada como início para eliminar rotações equivalentes.
    O algoritmo usa tempo O(n² 2ⁿ) e memória O(n 2ⁿ).
    """
    quantidade_cidades = len(distancias)
    outras_cidades = list(range(1, quantidade_cidades))

    # custos[(mascara, destino)] guarda o menor custo do caminho que sai de 0,
    # visita exatamente as cidades da máscara e termina em destino.
    custos = {}
    for cidade in outras_cidades:
        mascara = criar_mascara([cidade])
        custos[(mascara, cidade)] = distancias[0][cidade]

    for tamanho in range(2, quantidade_cidades):
        for subconjunto in itertools.combinations(outras_cidades, tamanho):
            mascara = criar_mascara(subconjunto)
            for destino in subconjunto:
                mascara_anterior = mascara ^ (1 << (destino - 1))
                possibilidades = []
                for anterior in subconjunto:
                    if anterior != destino:
                        custo = (
                            custos[(mascara_anterior, anterior)]
                            + distancias[anterior][destino]
                        )
                        possibilidades.append(custo)
                custos[(mascara, destino)] = min(possibilidades)

    mascara_completa = (1 << (quantidade_cidades - 1)) - 1
    ultima_cidade = min(
        outras_cidades,
        key=lambda cidade: custos[(mascara_completa, cidade)]
        + distancias[cidade][0],
    )
    custo_otimo = (
        custos[(mascara_completa, ultima_cidade)]
        + distancias[ultima_cidade][0]
    )

    # Reconstrói a rota seguindo, em cada estado, o predecessor de menor custo.
    caminho_invertido = []
    mascara = mascara_completa
    destino = ultima_cidade
    while mascara:
        caminho_invertido.append(destino)
        mascara_anterior = mascara ^ (1 << (destino - 1))
        if mascara_anterior == 0:
            break
        candidatos = [
            cidade
            for cidade in outras_cidades
            if mascara_anterior & (1 << (cidade - 1))
        ]
        destino = min(
            candidatos,
            key=lambda cidade: custos[(mascara_anterior, cidade)]
            + distancias[cidade][destino],
        )
        mascara = mascara_anterior

    rota_otima = [0] + list(reversed(caminho_invertido))
    return rota_otima, custo_otimo
