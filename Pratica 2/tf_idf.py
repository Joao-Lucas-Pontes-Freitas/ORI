import re

import numpy as np
from unidecode import unidecode


def produto_escalar(vetor1, vetor2):
    return sum(a * b for a, b in zip(vetor1, vetor2))


def similaridade(vetor1, vetor2):
    if not vetor1 or not vetor2:
        return 0

    numerador = produto_escalar(vetor1, vetor2)
    denominador = np.linalg.norm(vetor1)

    if denominador == 0:
        return 0

    return numerador / denominador


def palavras_documento(arquivo, vocabulario):
    with open(arquivo, encoding="utf-8") as f:
        texto = f.read()

        sem_acentos = unidecode(texto)
        somente_letras = re.sub(r"[^a-zA-Z ]", " ", sem_acentos)

        minusculas = somente_letras.lower()

        palavras = minusculas.split()

        if vocabulario:
            palavras = [p for p in palavras if p in vocabulario]

        return palavras


def tf(palavra, documento, vocabulario):
    palavras_doc = palavras_documento(documento, vocabulario)

    if not palavras_doc:
        return 0

    f = palavras_doc.count(palavra)

    if f == 0:
        return 0

    return 1 + np.log2(f)


def idf(palavra, documentos, vocabulario):
    n = len(documentos)

    if n == 0:
        return 0

    df = sum(1 for doc in documentos if tf(palavra, doc, vocabulario) > 0)

    return np.log2(n / df) if df > 0 else 0


def tf_idf(palavra, documentos, vocabulario):
    return [
        tf(palavra, doc, vocabulario) * idf(palavra, documentos, vocabulario)
        for doc in documentos
    ]


def similaridade_consulta(consulta, documentos, vocabulario):
    palavras_consulta = palavras_documento(consulta, vocabulario)
    vetor_consulta = [
        idf(palavra, documentos, vocabulario) if palavra in palavras_consulta else 0
        for palavra in vocabulario
    ]
    similaridades = []

    for documento in documentos:
        vetor_documento = [
            (
                tf_idf(palavra, documentos, vocabulario)[documentos.index(documento)]
                if palavra in palavras_consulta
                else 0
            )
            for palavra in vocabulario
        ]
        similaridade_doc = similaridade(vetor_documento, vetor_consulta)
        similaridades.append(similaridade_doc)

    return similaridades


if __name__ == "__main__":
    documentos = [
        "documents/d1.txt",
        "documents/d2.txt",
        "documents/d3.txt",
        "documents/d4.txt",
    ]
    vocabulario = [
        "to",
        "do",
        "is",
        "be",
        "or",
        "not",
        "i",
        "am",
        "what",
        "think",
        "therefore",
        "da",
        "let",
        "it",
    ]

    consulta = "consulta.txt"
    similaridades = similaridade_consulta(consulta, documentos, vocabulario)

    for doc, sim in zip(documentos, similaridades):
        print(f"Similaridade com {doc}: {sim:.4f}")
