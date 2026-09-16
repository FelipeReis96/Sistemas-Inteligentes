"""AG adaptado para rotas por permutação."""

import random

from problema import calcular_custo_rota


def gerar_populacao(cidades, tamanho_populacao, semente):
    """Cria uma lista de rotas aleatórias e independentes."""
    gerador = random.Random(semente)
    populacao = []
    for _ in range(tamanho_populacao):
        rota = list(range(len(cidades)))
        gerador.shuffle(rota)
        populacao.append(rota)
    return populacao


def avaliar_populacao(populacao, distancias):
    """Calcula o custo de cada rota, na mesma ordem da população."""
    custos = []
    for rota in populacao:
        custos.append(calcular_custo_rota(rota, distancias))
    return custos


def calcular_aptidoes(custos):
    """Transforma a minimização do custo em maximização da aptidão."""
    aptidoes = []
    for custo in custos:
        # O +1 também permite avaliar rotas com custo zero.
        aptidoes.append(1.0 / (1.0 + custo))
    return aptidoes


def selecionar_por_roleta(populacao, aptidoes, gerador):
    """Sorteia um pai com probabilidade proporcional à sua aptidão."""
    sorteio = gerador.random() * sum(aptidoes)
    acumulado = 0.0
    for i in range(len(populacao)):
        acumulado += aptidoes[i]
        if sorteio < acumulado:
            return populacao[i]
    return populacao[-1]


def criar_filho_pmx(pai_a, pai_b, inicio, fim):
    """Copia pai_b[inicio:fim] e resolve repetições por mapeamento."""
    filho = pai_a.copy()
    trecho_b = pai_b[inicio:fim]
    filho[inicio:fim] = trecho_b

    for posicao in range(len(filho)):
        if inicio <= posicao < fim:
            continue
        cidade = pai_a[posicao]
        # Se a cidade já está no trecho copiado, seguimos o par B -> A.
        # Pode ser necessário seguir mais de um par até eliminar o conflito.
        while cidade in trecho_b:
            posicao_mapeada = inicio + trecho_b.index(cidade)
            cidade = pai_a[posicao_mapeada]
        filho[posicao] = cidade
    return filho


def cruzar_pmx(pai_a, pai_b, gerador):
    """Gera dois filhos com os mesmos pontos de corte."""
    inicio, ultima = sorted(gerador.sample(range(len(pai_a)), 2))
    fim = ultima + 1
    filho_a = criar_filho_pmx(pai_a, pai_b, inicio, fim)
    filho_b = criar_filho_pmx(pai_b, pai_a, inicio, fim)
    return filho_a, filho_b


def mutar_por_troca(rota, taxa_mutacao, gerador):
    """Para cada posição, sorteia se ela será trocada com outra posição."""
    nova_rota = rota.copy()
    for posicao in range(len(nova_rota)):
        if gerador.random() < taxa_mutacao:
            outra = gerador.randrange(len(nova_rota) - 1)
            if outra >= posicao:
                outra += 1
            nova_rota[posicao], nova_rota[outra] = (
                nova_rota[outra], nova_rota[posicao]
            )
    return nova_rota


def gerar_descendentes(populacao, aptidoes, taxa_cruzamento, taxa_mutacao, gerador):
    """Produz tantos filhos quanto pais, selecionando com reposição."""
    descendentes = []
    while len(descendentes) < len(populacao):
        pai_a = selecionar_por_roleta(populacao, aptidoes, gerador)
        pai_b = selecionar_por_roleta(populacao, aptidoes, gerador)

        if gerador.random() < taxa_cruzamento:
            filhos = cruzar_pmx(pai_a, pai_b, gerador)
        else:
            filhos = [pai_a.copy(), pai_b.copy()]

        for filho in filhos:
            if len(descendentes) == len(populacao):
                break
            descendentes.append(mutar_por_troca(filho, taxa_mutacao, gerador))
    return descendentes


def selecionar_sobreviventes(populacao, custos, tamanho):
    """Retém as rotas de menor custo, reutilizando as avaliações feitas."""
    indices = list(range(len(populacao)))
    indices.sort(key=lambda indice: custos[indice])
    sobreviventes = []
    custos_sobreviventes = []
    for indice in indices[:tamanho]:
        sobreviventes.append(populacao[indice].copy())
        custos_sobreviventes.append(custos[indice])
    return sobreviventes, custos_sobreviventes


def executar_algoritmo_genetico(
    populacao_inicial, distancias, taxa_cruzamento, taxa_mutacao, geracoes, semente
):
    """Evolui uma população não vazia de rotas válidas por gerações completas.

    Os parâmetros são definidos pelo experimento: taxas entre 0 e 1 e número
    inteiro de gerações >= 0. A população inicial conta no total de avaliações.
    """
    gerador = random.Random(semente)
    tamanho = len(populacao_inicial)
    custos = avaliar_populacao(populacao_inicial, distancias)
    populacao, custos = selecionar_sobreviventes(populacao_inicial, custos, tamanho)
    melhor_custo_inicial = custos[0]
    quantidade_avaliacoes = tamanho
    historico_melhor_custo = [custos[0]]
    historico_avaliacoes = [quantidade_avaliacoes]

    for _ in range(geracoes):
        aptidoes = calcular_aptidoes(custos)
        filhos = gerar_descendentes(
            populacao, aptidoes, taxa_cruzamento, taxa_mutacao, gerador
        )
        custos_filhos = avaliar_populacao(filhos, distancias)
        quantidade_avaliacoes += len(filhos)

        # Conforme os slides: melhores entre pais e filhos formam a geração.
        populacao, custos = selecionar_sobreviventes(
            populacao + filhos, custos + custos_filhos, tamanho
        )
        historico_melhor_custo.append(custos[0])
        historico_avaliacoes.append(quantidade_avaliacoes)

    return {
        "melhor_rota": populacao[0].copy(),
        "melhor_custo": custos[0],
        "melhor_custo_inicial": melhor_custo_inicial,
        "populacao_final": populacao,
        "custos_finais": custos,
        "historico_melhor_custo": historico_melhor_custo,
        "historico_avaliacoes": historico_avaliacoes,
        "quantidade_avaliacoes": quantidade_avaliacoes,
    }
