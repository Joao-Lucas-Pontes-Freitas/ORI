from unidecode import unidecode
import re

def index(arquivo):
    texto = open(arquivo, encoding='utf8').read()

    sem_acentos = unidecode(texto)
    somente_letras = re.sub(r'[^a-zA-Z\s]', ' ', sem_acentos)

    minusculas = somente_letras.lower()
    unicas = set(minusculas.split())
    ordenado = sorted(unicas)

    return ordenado

if __name__ == '__main__':
    palavras = index(arquivo='entrada.txt')
    for palavra in palavras:
        print(palavra)

    # Caso de teste
    # esperado = ["amado", "brasileiro", "clube", "dentre", "es", "forte", "grande", "grandes", "o", "os", "paulista", "primeiro", "salve", "tricolor", "tu"]
    # print(palavras == esperado)
