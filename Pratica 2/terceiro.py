import numpy as np
from collections import Counter

def construir_vocabulario(documentos, consulta):
    """
    Constrói um vocabulário ordenado a partir dos documentos e da consulta.
    """
    termos = {termo for doc in documentos for termo in doc} | set(consulta)
    return sorted(termos)

def calcular_matriz_tf(documentos, vocab):
    """
    Calcula a matriz TF (termo×documento) densamente:
    tf = 1 + log₂(frequência) se a frequência > 0, senão 0.
    Retorna um ndarray de forma (V, D).
    """
    V = len(vocab)
    D = len(documentos)
    indice = {termo: i for i, termo in enumerate(vocab)}
    contadores = [Counter(doc) for doc in documentos]

    tf = np.zeros((V, D), dtype=float)
    for j, contador in enumerate(contadores):
        for termo, freq in contador.items():
            i = indice[termo]
            tf[i, j] = 1 + np.log2(freq)
    return tf

def calcular_vetor_idf(tf_matrix):
    """
    Calcula o vetor IDF de forma vetorizada:
    idf[i] = log₂(N / df_i), onde df_i = número de documentos contendo o termo i.
    """
    V, D = tf_matrix.shape
    df = np.count_nonzero(tf_matrix > 0, axis=1)
    df = np.where(df > 0, df, 1)   # evita divisão por zero
    idf = np.log2(D / df)
    return idf

def aplicar_pesos_tfidf(tf_matrix, idf_vector):
    """
    Aplica o peso TF-IDF multiplicando cada linha de TF pelo IDF correspondente.
    Retorna matriz de forma (V, D).
    """
    return tf_matrix * idf_vector[:, None]

def calcular_vetor_consulta(consulta, vocab, idf_vector):
    """
    Constrói o vetor TF-IDF da consulta no mesmo espaço do vocabulário.
    """
    V = len(vocab)
    indice = {termo: i for i, termo in enumerate(vocab)}
    cont_q = Counter(consulta)

    q_tf = np.zeros(V, dtype=float)
    for termo, freq in cont_q.items():
        if termo in indice:
            q_tf[indice[termo]] = 1 + np.log2(freq)

    q_tfidf = q_tf * idf_vector
    return q_tfidf

def calcular_similaridades_cosseno(tfidf_matrix, vetor_consulta):
    """
    Calcula similaridade de cosseno entre cada coluna de tfidf_matrix e vetor_consulta.
    Retorna um array de tamanho D.
    """
    normas_docs  = np.linalg.norm(tfidf_matrix, axis=0)
    norma_consulta = np.linalg.norm(vetor_consulta)
    denom = normas_docs * norma_consulta + 1e-9  # para evitar divisão por zero

    return (tfidf_matrix.T @ vetor_consulta) / denom

def obter_similaridades(consulta, documentos):
    """
    Função principal:
    Dada uma consulta (lista de tokens) e documentos (lista de listas de tokens),
    retorna as similaridades de cosseno entre consulta e cada documento.
    """
    # 1) Vocabulário
    vocab = construir_vocabulario(documentos, consulta)

    # 2) TF
    tf = calcular_matriz_tf(documentos, vocab)

    # 3) IDF
    idf = calcular_vetor_idf(tf)

    # 4) TF-IDF dos documentos
    tfidf_docs = aplicar_pesos_tfidf(tf, idf)

    # 5) TF-IDF da consulta
    vetor_consulta = calcular_vetor_consulta(consulta, vocab, idf)

    # 6) Similaridades
    sims = calcular_similaridades_cosseno(tfidf_docs, vetor_consulta)
    return sims.tolist()


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

    similaridades = obter_similaridades(consulta, documentos)
    for idx, score in enumerate(similaridades, start=1):
        print(f"Documento {idx}: similaridade = {score:.6f}")
