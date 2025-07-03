import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin

class Log2TfidfTransformer(TransformerMixin):
    """
    TF-IDF custom:
      tf  = 1 + log₂(count)
      idf = log₂(N / df)
    Sem smoothing e sem normalização interna.
    """
    def __init__(self):
        self.idf_ = None

    def fit(self, X, y=None):
        # X: termo-documento (n_docs×n_terms)
        df = np.array((X > 0).sum(axis=0)).ravel()
        N  = X.shape[0]
        df[df == 0] = 1
        self.idf_ = np.log2(N / df)
        return self

    def transform(self, X):
        Xc = X.copy()
        Xc.data = 1 + np.log2(Xc.data)
        return Xc.multiply(self.idf_)


def manual_sparse_tfidf(consulta, documentos):
    # 1) vocabulário e índice
    vocab = sorted({w for doc in documentos for w in doc} | set(consulta))
    idx   = {w: i for i, w in enumerate(vocab)}
    V, D  = len(vocab), len(documentos)

    # 2) monta TF esparso (termo×doc)
    rows, cols, data = [], [], []
    for j, doc in enumerate(documentos):
        cont = {}
        for w in doc:
            cont[w] = cont.get(w, 0) + 1
        for w, f in cont.items():
            i = idx[w]
            rows.append(i); cols.append(j)
            data.append(1 + np.log2(f))
    tf = csr_matrix((data, (rows, cols)), shape=(V, D))

    # 3) calcula IDF
    df  = np.array((tf > 0).sum(axis=1)).ravel()
    idf = np.log2(D / (df + 1e-9))

    # 4) TF-IDF dos docs e norma de cada doc
    docs_tfidf = tf.multiply(idf[:, None])
    norms_d    = np.linalg.norm(docs_tfidf.toarray(), axis=0)

    # 5) monta vetor consulta
    cont_q, q_rows, q_data = {}, [], []
    for w in consulta:
        if w in idx:
            cont_q[w] = cont_q.get(w, 0) + 1
    for w, f in cont_q.items():
        i = idx[w]; q_rows.append(i)
        q_data.append(1 + np.log2(f))
    q = csr_matrix((q_data, (q_rows, [0]*len(q_rows))), shape=(V, 1))
    q = q.multiply(idf[:, None])
    q_dense = q.toarray().ravel()
    norm_q  = np.linalg.norm(q_dense)

    # 6) similaridade cosseno exata
    sims = (docs_tfidf.T @ q).toarray().ravel() / (norms_d * norm_q + 1e-9)
    return sims.tolist()


def sklearn_tfidf_custom(consulta, documentos):
    # 1) stringify
    docs_str = [" ".join(doc) for doc in documentos]
    q_str    = [" ".join(consulta)]

    # 2) CountVectorizer → contagens
    vect   = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X_docs = vect.fit_transform(docs_str)
    X_q    = vect.transform(q_str)

    # 3) TF-IDF custom log₂
    tfidf       = Log2TfidfTransformer().fit(X_docs)
    Xd_tfidf_csr = tfidf.transform(X_docs)   # shape (n_docs, n_terms)
    Xq_tfidf_csr = tfidf.transform(X_q)      # shape (1, n_terms)

    # 4) converte p/ denso para normalizar igual ao manual
    docs_dense = Xd_tfidf_csr.toarray().T    # (n_terms, n_docs)
    q_dense    = Xq_tfidf_csr.toarray().ravel()
    norms_d    = np.linalg.norm(docs_dense, axis=0)
    norm_q     = np.linalg.norm(q_dense)

    # 5) similaridade cosseno exata
    sims = (docs_dense.T @ q_dense) / (norms_d * norm_q + 1e-9)
    return sims.tolist()


if __name__ == "__main__":
    consulta = ["to", "do"]
    documentos = [
        ["to","do","is","to","be","to","be","is","to","do"],
        ["to","be","or","not","to","be","i","am","what","i","am"],
        ["i","think","therefore","i","am","do","be","do","be","do"],
        ["do","do","do","da","da","da","let","it","be","let","it","be"],
    ]

    print("manual sparse:", manual_sparse_tfidf(consulta, documentos))
    print("sklearn custom:", sklearn_tfidf_custom(consulta, documentos))