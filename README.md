# Sistemas Inteligentes

## Entrega 1

Problema do Caixeiro Viajante com Têmpera Simulada e Algoritmo Genético.
O artigo está em `tarefa-1/artigo.pdf` e o código usa apenas a biblioteca
padrão do Python.

Execute a partir da raiz do repositório:

```sh
python3 tarefa-1/src/main.py --algorithm tempera --cities 40
python3 tarefa-1/src/main.py --algorithm ag --cities 40
```

## Entrega 2

Classificação e regressão com Random Forest e MLP usando sinais vitais.

Na raiz do repositório, prepare o ambiente:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tarefa-2/requirements.txt
```

Execute nesta ordem: exploração dos dados, experimentos, treinamento e
avaliação, e geração das figuras do artigo.

```sh
.venv/bin/python tarefa-2/src/explore.py
.venv/bin/python tarefa-2/src/experimentos.py
.venv/bin/python tarefa-2/src/train.py
.venv/bin/python tarefa-2/src/figuras_artigo.py
```

Os resultados da MLP podem variar entre ambientes, mesmo com a
mesma semente.
