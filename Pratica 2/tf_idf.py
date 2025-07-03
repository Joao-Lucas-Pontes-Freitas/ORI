import numpy as np


def similaridades_consulta(consulta, documentos, vocabulario):
    vetor_consulta = np.zeros(len(vocabulario))
    for i, palavra in enumerate(vocabulario):
        if palavra in consulta:
            vetor_consulta[i] = calcular_tf_idf(palavra, consulta, documentos)

    similaridades = []

    for documento in documentos:
        vetor_documento = np.zeros(len(vocabulario))
        for i, palavra in enumerate(vocabulario):
            if palavra in documento:
                vetor_documento[i] = calcular_tf_idf(palavra, documento, documentos)
        similaridade = calcular_similaridade(vetor_consulta, vetor_documento)
        similaridades.append(similaridade)

    return similaridades


def calcular_tf_idf(palavra, documento, documentos):
    tf = calcular_tf(palavra, documento)
    idf = calcular_idf(palavra, documentos)

    return tf * idf


def calcular_tf(palavra, documento):
    if not documento:
        return 0

    f = documento.count(palavra)

    if f == 0:
        return 0

    return 1 + np.log2(f)


def calcular_idf(palavra, documentos):
    if not documentos:
        return 0

    ni = sum(1 for doc in documentos if palavra in doc)

    return np.log2(len(documentos) / (ni))


def calcular_similaridade(vetor_consulta, vetor_documento):
    numerador = np.dot(vetor_consulta, vetor_documento)
    denominador = np.linalg.norm(vetor_documento)

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

    similaridades = similaridades_consulta(consulta, documentos, vocabulario)

    for i, sim in enumerate(similaridades):
        print(f"Similaridade do documento {i+1}: {sim:.3f}")
