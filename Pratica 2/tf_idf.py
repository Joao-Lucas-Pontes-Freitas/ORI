import numpy as np


def similaridades_consulta(consulta, documentos, vocabulario):
    matriz_tf, matriz_idf = matriz_tf_idf(documentos, vocabulario)
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


def calcular_idf(palavra, documentos, matriz_tf):
    N = len(documentos)
    ni = np.count_nonzero(matriz_tf[palavra, :])
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
    denominador = np.linalg.norm(vetor_documento) * np.linalg.norm(vetor_consulta)

    if denominador == 0:
        return 0

    return numerador / denominador


if __name__ == "__main__":
    consulta = ["to", "do"]
    documentos = [
        # d1: "To do is to be. To be is to do."
        ["to", "do", "is", "to", "be", "to", "be", "is", "to", "do"],
        # d2: "To be or not to be. I am what I am."
        ["to", "be", "or", "not", "to", "be", "i", "am", "what", "i", "am"],
        # d3: "I think therefore I am. Do be do be do."
        ["i", "think", "therefore", "i", "am", "do", "be", "do", "be", "do"],
        # d4: "Do do do, da da da. Let it be, let it be."
        ["do", "do", "do", "da", "da", "da", "let", "it", "be", "let", "it", "be"],
    ]

    vocabulario = set(palavra for doc in documentos for palavra in doc)
    vocabulario.update(consulta)

    similaridades_2 = similaridades_consulta(consulta, documentos, vocabulario)

    for i, sim in enumerate(similaridades_2):
        print(f"Similaridade do documento {i+1}: {sim:.3f}")
