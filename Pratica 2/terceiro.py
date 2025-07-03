from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix


def construir_vocabulario(documentos, consulta):
    """
    Constrói um vocabulário ordenado sem duplicatas.
    """
    termos = set()
    for doc in documentos:
        termos.update(doc)
    for termo in consulta:
        if termo not in termos:
            termos.add(termo)
    return sorted(termos)


def construir_matriz_tf_csr(documentos, vocab):
    """
    Constrói matriz TF esparsa (CSR) de forma eficiente:
    tf = 1 + log₂(frequência) para f > 0.
    """
    idx = {termo: i for i, termo in enumerate(vocab)}
    V = len(vocab)
    D = len(documentos)
    rows, cols, data = [], [], []
    for j, doc in enumerate(documentos):
        counter = Counter(doc)
        for termo, freq in counter.items():
            i = idx[termo]
            rows.append(i)
            cols.append(j)
            data.append(1 + np.log2(freq))
    return csr_matrix((data, (rows, cols)), shape=(V, D))


def calcular_idf(idf_matrix):
    """
    Recebe matriz TF esparsa e retorna vetor IDF:
    idf[i] = log₂(N / df_i).
    """
    V, D = idf_matrix.shape
    # conta em quantos documentos cada termo aparece
    df = np.ravel(idf_matrix.astype(bool).sum(axis=1))
    df = np.where(df > 0, df, 1)  # evita divisão por zero
    return np.log2(D / df)


def aplicar_tfidf_csr(tf_csr, idf_vector):
    """
    Multiplica cada linha (termo) pelo seu idf, mantendo CSR.
    """
    return tf_csr.multiply(idf_vector[:, None])


def montar_vetor_consulta_csr(consulta, vocab, idf_vector):
    """
    Constrói vetor TF-IDF da consulta como array denso,
    pois geralmente é pequeno.
    """
    idx = {termo: i for i, termo in enumerate(vocab)}
    V = len(vocab)
    q_tf = np.zeros(V, dtype=float)
    counter = Counter(consulta)
    for termo, freq in counter.items():
        if termo in idx:
            q_tf[idx[termo]] = 1 + np.log2(freq)
    return q_tf * idf_vector


def calcular_similaridades_cosseno_sparse(docs_tfidf_csr, q_tfidf):
    """
    Calcula similaridades de cosseno entre cada coluna de docs_tfidf_csr
    e o vetor denso q_tfidf.
    """
    # normas dos documentos: sqrt(sum de quadrados por coluna)
    sq = docs_tfidf_csr.power(2)
    norms_docs = np.sqrt(np.ravel(sq.sum(axis=0)))
    norm_q = np.linalg.norm(q_tfidf)
    # produto interno: cada coluna ⋅ vetor consulta
    # docs_tfidf_csr.T é (D, V), q_tfidf é (V,)
    dots = docs_tfidf_csr.T.dot(q_tfidf)
    return np.array(dots / (norms_docs * norm_q + 1e-9))


def obter_similaridades(consulta, documentos):
    """
    Pipeline completo sem scikit-learn, usando scipy.sparse.
    Retorna lista de similaridades de cosseno.
    """
    # 1) vocabulário
    vocab = construir_vocabulario(documentos, consulta)
    # 2) TF esparso
    tf_csr = construir_matriz_tf_csr(documentos, vocab)
    # 3) IDF
    idf = calcular_idf(tf_csr)
    # 4) TF-IDF esparso
    docs_tfidf = aplicar_tfidf_csr(tf_csr, idf)
    # 5) vetor consulta
    q_tfidf = montar_vetor_consulta_csr(consulta, vocab, idf)
    # 6) similaridades
    sims = calcular_similaridades_cosseno_sparse(docs_tfidf, q_tfidf)
    return sims.tolist()


if __name__ == "__main__":
    consulta = ["to", "do"]
    documentos = [
        ["to", "do", "is", "to", "be", "to", "be", "is", "to", "do"],
        ["to", "be", "or", "not", "to", "be", "i", "am", "what", "i", "am"],
        ["i", "think", "therefore", "i", "am", "do", "be", "do", "be", "do"],
        ["do", "do", "do", "da", "da", "da", "let", "it", "be", "let", "it", "be"],
    ]
    similaridades = obter_similaridades(consulta, documentos)
    for idx, score in enumerate(similaridades, start=1):
        print(f"Documento {idx}: similaridade = {score:.6f}")
