import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

from exercicio1 import exercicio1


def criar_vocabulario(df, text_column="Traduzido"):
    """
    Cria um vocabulário único a partir da coluna de texto especificada.
    Salva o vocabulário em 'vocabulario.txt'.
    """
    print("Criando vocabulário único")

    vocabulario = set()
    for texto in df[text_column].dropna():
        palavras = texto.split()
        vocabulario.update(palavras)

    with open("vocabulario.txt", "w", encoding="utf-8") as f:
        for palavra in sorted(vocabulario):
            f.write(f"{palavra}\n")

    print(f"Vocabulário salvo em 'vocabulario.txt' com {len(vocabulario)} palavras")

    return sorted(vocabulario)


def tf_idf(df, vocabulario, text_column="Traduzido"):
    """
    Calcula a matriz TF-IDF para os textos na coluna especificada.
    Retorna um DataFrame com a matriz TF-IDF.
    """

    print("Calculando matriz TF-IDF")
    vec_fit = TfidfVectorizer(vocabulary=vocabulario, stop_words="english")
    X = vec_fit.fit_transform(df[text_column].fillna("").astype(str))

    matriz = pd.DataFrame(X.toarray(), columns=vec_fit.get_feature_names_out())

    with open("matriz_tfidf.csv", "w", encoding="utf-8") as f:
        matriz.to_csv(f, index=False)

    print(f"Matriz TF-IDF salva em 'matriz_tfidf.csv'")


def treinar_modelo(
    df, vocabulario, text_column="Traduzido", label_column="Classificação"
):
    """
    Treina e avalia modelo MultinomialNB com validação cruzada.
    Imprime acurácia média, matriz de confusão e relatório de classificação.
    """

    print("Treinando modelo")
    X = df[text_column].astype(str)
    y = df[label_column].astype(str).str.lower().str.strip()

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(vocabulary=vocabulario, stop_words="english")),
            ("nb", MultinomialNB()),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print("Avaliando o modelo")
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    return scores, y, y_pred


def avaliar_modelo(y, y_pred, scores):
    print()
    print(f"Acurácia média (CV): {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Acurácia geral (CV predict): {accuracy_score(y, y_pred):.4f}")
    print()
    print("Matriz de confusão [negativo, neutro, positivo]:")
    matriz = confusion_matrix(y, y_pred, labels=["negativo", "neutro", "positivo"])
    print(matriz)
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=["negativo", "neutro", "positivo"], yticklabels=["negativo", "neutro", "positivo"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()

    print()
    print("Relatório:")
    print(
        classification_report(
            y, y_pred, labels=["negativo", "neutro", "positivo"], digits=3
        )
    )


def exercicio3():
    df = exercicio1()
    vocabulario = criar_vocabulario(df, text_column="Traduzido")
    tf_idf(df, vocabulario, text_column="Traduzido")
    scores, y, y_pred = treinar_modelo(df, vocabulario, label_column="Classificação")
    avaliar_modelo(y, y_pred, scores)


if __name__ == "__main__":
    exercicio3()
