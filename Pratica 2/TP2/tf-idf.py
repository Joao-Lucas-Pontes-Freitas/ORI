import os
import time
import unicodedata

import numpy as np


def similaridades_consulta(consulta, documentos, vocabulario, matriz_tf, matriz_idf):
    vetor_consulta = np.zeros(len(vocabulario))

    for i, palavra in enumerate(vocabulario):
        vetor_consulta[i] = matriz_idf[i] if palavra in consulta else 0

    similaridades = []

    for j in range(len(documentos)):
        vetor_documento = matriz_tf[:, j] * matriz_idf
        similaridade = calcular_similaridade(vetor_consulta, vetor_documento)
        similaridades.append(similaridade)

    return similaridades


def matriz_tf_idf(documentos, vocabulario):
    matriz_tf = np.zeros((len(vocabulario), len(documentos)))

    for i, palavra in enumerate(vocabulario):
        for j, documento in enumerate(documentos):
            matriz_tf[i][j] = calcular_tf(palavra, documento)

    matriz_idf = np.zeros(len(vocabulario))

    for i, palavra in enumerate(vocabulario):
        matriz_idf[i] = calcular_idf(i, documentos, matriz_tf)

    return matriz_tf, matriz_idf


def calcular_idf(idx_palavra, documentos, matriz_tf):
    N = len(documentos)
    ni = np.count_nonzero(matriz_tf[idx_palavra, :])
    if ni == 0:
        return 0
    return np.log2(N / ni)


def calcular_tf(palavra, documento):
    if not documento:
        return 0

    f = documento.count(palavra)

    if f == 0:
        return 0

    return 1 + np.log2(f)


def calcular_similaridade(vetor_consulta, vetor_documento):
    numerador = np.dot(vetor_consulta, vetor_documento)
    denominadores = [
        np.linalg.norm(vetor_documento)
        * np.linalg.norm(vetor_consulta),  # usando fórmula do cosseno completa
        np.linalg.norm(vetor_documento),
    ]

    return [0 if d == 0 else numerador / d for d in denominadores]


def normalizar_texto(documento):
    t = "".join(documento).lower()
    nfkd = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in nfkd if not unicodedata.combining(c))
    t = "".join(c for c in t if c.isalpha() or c.isspace())
    return t.split()


def ler_documentos(arquivo):
    with open(arquivo, "r", encoding="utf-8") as f:
        return " ".join(linha.strip() for linha in f.readlines())


def get_paths(diretorio):
    caminhos = []
    for root, _, files in os.walk(diretorio):
        for nome in files:
            caminhos.append(os.path.join(root, nome))
    return caminhos


if __name__ == "__main__":

    start_time = time.time()

    caminhos = get_paths("../docs_clean")
    nomes = [os.path.basename(caminho) for caminho in caminhos]
    documentos = [ler_documentos(caminho) for caminho in caminhos]

    documentos = [normalizar_texto(doc) for doc in documentos]
    vocabulario = sorted(set(palavra for doc in documentos for palavra in doc))

    finish_time = time.time()
    time_spent = finish_time - start_time

    print()
    # print("Vocabulário:")
    # print(vocabulario)
    # print()
    print("Tamanho do vocabulário:", len(vocabulario))
    print(f"Tempo gasto na leitura, normalização dos documentos e criação do vocabulário: {time_spent} segundos")
    print()

    start_time = time.time()

    matriz_tf, matriz_idf = matriz_tf_idf(documentos, vocabulario)
    pesos_tf_idf = matriz_tf * matriz_idf[:, np.newaxis]

    finish_time = time.time()
    time_spent = finish_time - start_time

    best_index = pesos_tf_idf.argmax()
    palavra_idx, doc_idx = np.unravel_index(best_index, pesos_tf_idf.shape)

    print("Matriz TF-IDF:")
    print(pesos_tf_idf)
    print()

    print(f"A palavra que teve maior peso TF-IDF foi '{vocabulario[palavra_idx]}' no documento '{nomes[doc_idx]}'")
    print("O valor de TF-IDF foi de:", pesos_tf_idf[palavra_idx, doc_idx])
    print(f"Tempo gasto no processamento das matrizes TF e IDF: {time_spent} segundos")
    print()

    consulta = " ".join(input("Digite a consulta ou enter para sair: ").strip().split())
    while consulta:
        start_time = time.time()
        consulta_normalizada = normalizar_texto(consulta)
        similaridades = similaridades_consulta(consulta_normalizada, documentos, vocabulario, matriz_tf, matriz_idf)
        finish_time = time.time()

        time_spent = finish_time - start_time

        print()
        print(f"Tempo gasto na consulta: {time_spent} segundos")
        print()
        print("Similaridades da consulta com os documentos:")
        print()

        resultados = [(i, nomes[i], similaridades[i]) for i in range(len(similaridades))]
        resultados_ordenados = sorted(resultados, key=lambda x: x[2][0], reverse=True)

        for i, nome, similaridade in resultados_ordenados:
            print(f"{nome} - {similaridade[0]:.4f}".replace('.', ','))

        print()

        consulta = " ".join(input("Digite a consulta ou enter para sair: ").strip().split())