import os
import re

import pandas as pd
import unidecode
from deep_translator import GoogleTranslator
from textblob import TextBlob


def limpa_texto(text):
    text = text.lower()  # minúsculas
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = unidecode.unidecode(text)  # tira acentos
    text = re.sub(r"[^a-z\s]", "", text)  # filtra apenas letras e espaços
    text = re.sub(r"\s+", " ", text).strip()  # remove espaços extras
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove emojis
    return text


def preprocess(path, text_column="Tweet", sep=","):
    """
    Pré-processa tweets: minúsculas, remove URLs, pontuação, números, acentos e emojis.
    """
    print(f"Iniciando pré-processamento do arquivo: {path}")

    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo CSV: {e}")

    if text_column not in df.columns:
        raise ValueError(f"A coluna '{text_column}' não foi encontrada no arquivo CSV.")

    df[text_column] = df[text_column].apply(limpa_texto)
    df = df[df[text_column].str.len() > 0]

    return df


def translate(
    dados,
    text_column="Tweet",
    file="tweets_traduzidos.csv",
    src_language="pt",
    target_language="en",
):
    """
    Traduz os textos da coluna especificada apenas uma vez.
    Salva todas as traduções em 'arquivo_traducao'.
    Nas próximas execuções, usa esse arquivo para evitar retradução.
    Retorna DataFrame com coluna 'Traduzido'.
    """
    dados = dados.copy()
    dados[text_column] = dados[text_column].astype(str)

    print("Iniciando tradução dos textos")

    # Carregar arquivo com traduções existentes
    if os.path.exists(file):
        traducoes = pd.read_csv(file)
        if set([text_column, "Traduzido"]) - set(traducoes.columns):
            traducoes = pd.DataFrame(columns=[text_column, "Traduzido"])
    else:
        traducoes = pd.DataFrame(columns=[text_column, "Traduzido"])

    # Garantir tipos e remover duplicatas
    traducoes[text_column] = traducoes[text_column].astype(str)
    traducoes = traducoes.dropna(subset=[text_column]).drop_duplicates(
        subset=[text_column], keep="last"
    )

    # Juntar traduções já feitas com os dados
    resultado = dados.merge(traducoes, on=text_column, how="left")

    # Encontrar textos sem tradução
    faltando = resultado.index[resultado["Traduzido"].isna()]
    if len(faltando) == 0:
        print("Todas as traduções já existem. Nada a traduzir.")
        return resultado

    # Traduzir apenas os que faltam
    tradutor = GoogleTranslator(source=src_language, target=target_language)
    novas = []

    for idx in faltando:
        texto = resultado.at[idx, text_column]
        try:
            traduzido = tradutor.translate(texto)
            traduzido = limpa_texto(traduzido)
        except Exception as e:
            print(f"Erro ao traduzir: {e}. Mantendo original.")
            traduzido = texto
        resultado.at[idx, "Traduzido"] = traduzido
        novas.append((texto, traduzido))

    # Atualizar e salvar todas as traduções
    banco_completo = pd.concat(
        [traducoes, pd.DataFrame(novas, columns=[text_column, "Traduzido"])],
        ignore_index=True,
    ).drop_duplicates(subset=[text_column], keep="last")

    banco_completo.to_csv(file, index=False)
    print(f"Traduções salvas em '{file}'")

    return resultado


def classify_sentiment(
    df,
    text_column="Tweet",
    translate_column="Traduzido",
    out_csv="sentiment_analysis.csv",
):
    """Calcula polaridade com TextBlob e salva [PT, EN, Sentiment]."""
    if translate_column not in df.columns:
        raise ValueError(f"Coluna '{translate_column}' não encontrada.")
    if text_column not in df.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada.")

    print("Iniciando análise de sentimento")

    def get_sentiment(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception as e:
            print(f"Erro ao analisar: {e}")
            return 0.0

    df["Sentiment"] = df[translate_column].apply(get_sentiment)
    df[[text_column, translate_column, "Sentiment"]].to_csv(out_csv, index=False)
    print(f"Resultados salvos em {out_csv}")
    return df


def exercicio1():
    """
    Executa o fluxo completo: pré-processamento, tradução e classificação de sentimento.
    """
    # Caminho para o arquivo CSV de entrada
    input_path = "reforma_previdencia_rotulado.csv"
    df = preprocess(input_path, text_column="Tweet", sep=";")  # sep do arquivo CSV
    df = translate(
        df, text_column="Tweet", src_language="pt", target_language="en"
    )  # traduz o texto
    df = classify_sentiment(df, text_column="Tweet", translate_column="Traduzido")
    return df


if __name__ == "__main__":
    exercicio1()
