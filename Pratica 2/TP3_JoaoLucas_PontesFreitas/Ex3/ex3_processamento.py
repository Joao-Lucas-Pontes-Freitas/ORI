import unicodedata
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


def processamento(text):
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r"[A-Za-z']+")
    stemmer = PorterStemmer()

    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    tokens = tokenizer.tokenize(text)
    tokens = [t.replace("'", "") for t in tokens]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


nltk.download("stopwords")

docs_path = Path("docs")
dst_path = Path("docs_clean")
dst_path.mkdir(exist_ok=True)

for file in docs_path.iterdir():
    if file.is_file():
        text = file.read_text(encoding="utf-8")
        clean = processamento(text)
        (dst_path / file.name).write_text(clean, encoding="utf-8")