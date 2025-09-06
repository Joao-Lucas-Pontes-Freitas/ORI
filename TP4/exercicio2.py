import pandas as pd
from exercicio1 import exercicio1


def comparar_labels(df_com_scores, col_real="Classificação", col_score="Sentiment"):
    y_true = df_com_scores[col_real].astype(str).str.lower().str.strip()

    def score_to_label(s):
        if s >= 0.5:
            return "positivo"
        if s <= -0.5:
            return "negativo"
        return "neutro"

    y_pred = df_com_scores[col_score].apply(score_to_label)

    matriz = pd.crosstab(
        y_true, y_pred, rownames=["Real"], colnames=["Predito"], dropna=False
    )

    print()

    for classe in ["negativo", "neutro", "positivo"]:
        total = matriz.loc[classe].sum() if classe in matriz.index else 0
        acertos = (
            matriz.loc[classe, classe]
            if (classe in matriz.index and classe in matriz.columns)
            else 0
        )
        erro_pct = 100 * (1 - acertos / total) if total > 0 else 0
        print(f"Erro {classe}: {erro_pct:.2f}% (acertos={acertos}, total={total})")

    print("\nMatriz de Confusão:")
    print(matriz)

def exercicio2():
    comparar_labels(exercicio1(), col_real="Classificação", col_score="Sentiment")

if __name__ == "__main__":
    exercicio2()