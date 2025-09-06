import time
import unicodedata
from pathlib import Path
import os

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


def similaridades_consulta(consulta, documentos, vocabulario, pesos_tf_idf, matriz_idf):
    vetor_consulta = np.zeros(len(vocabulario))
    consulta_set = set(consulta.split())

    for i, palavra in enumerate(vocabulario):
        vetor_consulta[i] = matriz_idf[i] if palavra in consulta_set else 0

    similaridades = []

    for j in range(pesos_tf_idf.shape[1]):
        vetor_documento = pesos_tf_idf[:, j]
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

    tokens = documento.split()
    f = sum(1 for t in tokens if t == palavra)

    return 0 if f == 0 else 1 + np.log2(f)


def calcular_similaridade(vetor_consulta, vetor_documento):
    numerador = np.dot(vetor_consulta, vetor_documento)
    denominador = np.linalg.norm(vetor_documento) * np.linalg.norm(vetor_consulta)

    return 0 if denominador == 0 else numerador / denominador


def processamento(text):
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r"[A-Za-z']+")
    stemmer = PorterStemmer()

    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    tokens = tokenizer.tokenize(text)
    tokens = [t.replace("'", "") for t in tokens]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


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

    caminhos = get_paths("docs_clean")
    nomes = [os.path.basename(caminho) for caminho in caminhos]
    documentos = [ler_documentos(caminho) for caminho in caminhos]
    documentos = [processamento(doc) for doc in documentos]

    print(f"Total de documentos: {len(documentos)}")

    vocabulario = sorted({tok for doc in documentos for tok in doc.split()})

    finish_time = time.time()
    time_spent = finish_time - start_time

    print()

    start_time = time.time()

    matriz_tf, matriz_idf = matriz_tf_idf(documentos, vocabulario)
    pesos_tf_idf = matriz_tf * matriz_idf[:, np.newaxis]

    finish_time = time.time()
    time_spent = finish_time - start_time

    best_index = pesos_tf_idf.argmax()
    palavra_idx, doc_idx = np.unravel_index(best_index, pesos_tf_idf.shape)

    print(
        f"A palavra que teve maior peso TF-IDF foi '{vocabulario[palavra_idx]}' no documento '{nomes[doc_idx]}'"
    )
    print("O valor de TF-IDF foi de:", pesos_tf_idf[palavra_idx, doc_idx])
    print(f"Tempo gasto no processamento das matrizes TF e IDF: {time_spent} segundos")
    print()

    consulta = " ".join(input("Digite a consulta ou enter para sair: ").strip().split())
    tempos_consulta = []

    while consulta:
        start_time = time.time()
        consulta_processada = processamento(consulta)
        print("Consulta processada:", consulta_processada)
        similaridades = similaridades_consulta(
            consulta_processada, documentos, vocabulario, pesos_tf_idf, matriz_idf
        )
        finish_time = time.time()

        time_spent = finish_time - start_time
        tempos_consulta.append(time_spent)

        print()
        print(f"Tempo gasto na consulta: {time_spent} segundos")
        print()
        print("Similaridades da consulta com os documentos:")
        print()

        resultados = [
            (i, nomes[i], similaridades[i]) for i in range(len(similaridades))
        ]
        resultados_ordenados = sorted(resultados, key=lambda x: x[2], reverse=True)

        for i, nome, similaridade in resultados_ordenados:
            print(f"{nome} - {similaridade:.4f}".replace('.txt', '').replace(".", ","))

        # for i, nome, similaridade in resultados:
        #     print(f"{similaridade:.4f}".replace(".txt", "").replace(".", ","))

        print()

        consulta = " ".join(
            input("Digite a consulta ou enter para sair: ").strip().split()
        )

    if len(tempos_consulta) > 0:
        print("Tempo médio gasto na consulta: {:.4f} segundos".format(sum(tempos_consulta) / len(tempos_consulta)))
