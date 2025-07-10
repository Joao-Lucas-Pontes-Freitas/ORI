import unicodedata

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize


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


def limpar_documentos(documentos, vocabulario):
    lemmatizer = WordNetLemmatizer()
    documentos_limpos = []
    for doc in documentos:
        tokens = wordpunct_tokenize(doc.lower())
        doc_limpo = []
        for palavra in tokens:
            if palavra.isalnum() and palavra in vocabulario:
                doc_limpo.append(lemmatizer.lemmatize(palavra))
        documentos_limpos.append(doc_limpo)
    return documentos_limpos


def normalizar_texto(textos):
    normalizado = []
    for texto in textos:
        t = texto.lower()
        t = t.replace("-", "")
        nfkd = unicodedata.normalize("NFKD", t)
        t = "".join(c for c in nfkd if not unicodedata.combining(c))
        normalizado.append(t)
    return normalizado


if __name__ == "__main__":
    # Caso de teste 1
    # consulta = "to do"
    # documentos = [
    #     "To do is to be. To be is to do.",
    #     "To be or not to be. I am what I am.",
    #     "I think therefore I am. Do be do be do.",
    #     "Do do do, da da da. Let it be, let it be.",
    # ]

    # vocabulario = []

    # Caso de teste 2
    # consulta = "dano prata caminhão"
    # documentos = [
    #     "navio ouro dano fogo",
    #     "entrega prata prata chegou caminhão",
    #     "navio, ouro chegou caminhão",
    # ]
    # vocabulario = [
    #     "navio",
    #     "ouro",
    #     "dano",
    #     "fogo",
    #     "entrega",
    #     "prata",
    #     "chegou",
    #     "caminhão",
    # ]

    # Exercicio
    consulta = "logan ororo x-men"
    documentos = [
        "logan e ororo são x-men",
        "stark, parker e logan já foram vingadores parker gostaria de ser novamente",
        "ororo e stark não são guardiões e sim vingadores, groot e rocket são guardiões mas poderiam ser vingadores",
        "eu sou groot logan todos somos groot o groot irá ajudar ororo e os x-men",
        "rocket e groot formam uma boa dupla nos guardiões rocket é maluco mas adora o groot",
    ]

    vocabulario = [
        "logan",
        "ororo",
        "stark",
        "parker",
        "groot",
        "rocket",
        "x-men",
        "vingadores",
        "guardiões",
    ]

    consulta = normalizar_texto([consulta])[0]
    documentos = normalizar_texto(documentos)
    vocabulario = normalizar_texto(vocabulario)

    if not vocabulario:
        vocabulario = set(wordpunct_tokenize(consulta.lower())) | {
            word for doc in documentos for word in wordpunct_tokenize(doc.lower())
        }

    documentos = limpar_documentos(documentos, vocabulario)
    consulta = limpar_documentos([consulta], vocabulario)[0]

    # for i, doc in enumerate(documentos):
    #     print(f"Documento {i+1} limpo: {doc}")

    # a)
    print()
    print("a)")
    matriz_tf, matriz_idf = matriz_tf_idf(documentos, vocabulario)
    print("Matriz TF:")
    print(matriz_tf)
    print()
    print("Matriz IDF:")
    print(matriz_idf)
    print()

    # b)
    print("b)")
    similaridades = similaridades_consulta(
        consulta, documentos, vocabulario, matriz_tf, matriz_idf
    )

    for i, sim in enumerate(similaridades):
        print(f"Similaridade do documento {i+1} usando cosseno: {sim[0]:.3f}")
        print(
            f"Similaridade do documento {i+1} usando somente norma do documento: {sim[1]:.3f}"
        )
        print()

    print("A consulta e o documento 1 são idênticos, o que faz sentido já que a similaridade é 1.0.")
    print("O documento 2 tem 1 termo da consulta, o que faz sentido já que a similaridade é próxima de 0.")
    print("O documento 3 tem 1 termo da consulta, o que faz sentido já que a similaridade é próxima de 0.")
    print("O documento 4 tem todos os termos da consulta, mas tem outros termos também, por isso a similaridade é menor que 1 mas é maior que as outras.")
    print("O documento 5 não tem nenhum termo da consulta, o que faz sentido já que a similaridade é 0.")
    print()

    # c)
    print("c)")
    print("Sim, é possível usar o modelo vetorial para comparar documentos. A consulta também pode ser vista como um documento já que é também somente um conjunto de termos.")
    similaridades = similaridades_consulta(
        documentos[0], documentos, vocabulario, matriz_tf, matriz_idf
    )

    print()

    print(f"Similaridade do documento 1 com documento 5: {similaridades[4][0]:.3f}")

    print("Isso faz total sentido, já que o documento 1 é idêntico à consulta e o documento 5 não tem nenhum termo da consulta, o que faz a similaridade ser 0.")

    print()