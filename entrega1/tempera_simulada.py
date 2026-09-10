import math
import random

NUM_MAX_ITERACOES = 1000
TEMP_FINAL = 0.1
TEMP_INICIAL = 1538 # temperatura de fusão do ferro
ALPHA = 0.9

# valores arbitrários, explicar depois


def funcao_aceitacao(temp, custo_diff):

    # Se tiver uma solução melhor, aceita na hora
    if custo_diff < 0:
        return True

    # Se a temperatura for menor ou igual a zero, não aceita (já está no estado final)
    if temp <= 0:
        return False
    
    p = math.exp(-custo_diff / temp)

    r = random.random()

    if r < p:
        return True
    return False


def atualizar_temperatura(temp0, alpha):
    return temp0 * alpha


# def calcular_custo(solucao): funcao aleatória para calcular o custo de uma solução
#     custo = solucao ** 2 + 2 * solucao + 1
#     return custo 


# gerar uma solução vizinha aleatória, por exemplo, adicionando ou subtraindo um valor aleatório à solução atual
# def gerar_solucao_vizinha(solucao): 
#     delta = random.uniform(-solucao, solucao)
#     return solucao + delta

# def tempera_simulada():

#     cont = 0
#     temp = TEMP_INICIAL
#     while cont < NUM_MAX_ITERACOES or temp <= TEMP_FINAL:

#         solucao_vizinha = gerar_solucao_vizinha(solucao_atual)
#         custo_vizinho = calcular_custo(solucao_vizinha)
#         custo_diff = custo_vizinho - custo_atual

#         if funcao_aceitacao(temp, custo_diff):
#             solucao_atual = solucao_vizinha
#             custo_atual = custo_vizinho
#         temp = atualizar_temperatura(temp, ALPHA)
#         cont += 1

