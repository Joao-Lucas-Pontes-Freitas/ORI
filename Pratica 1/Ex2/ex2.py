from unidecode import unidecode
import re

def index(arquivo):
    texto = open(arquivo, encoding='utf8').read()

    sem_acentos = unidecode(texto)
    somente_letras = re.sub(r'[^a-zA-Z ]', ' ', sem_acentos)

    minusculas = somente_letras.lower()
    unicas = set(minusculas.split())
    ordenado = sorted(unicas)

    return ordenado

def bag_of_words(documento, vocabulario):
    palavras_vocabulario = index(vocabulario)
    palavras_documento = index(documento)

    bag = {palavra: 0 for palavra in palavras_vocabulario}
    
    for palavra in bag:
        if palavra in palavras_documento and bag[palavra] == 0:
            bag[palavra] = 1

    return bag

if __name__ == '__main__':
    bag = bag_of_words(documento='entrada.txt', vocabulario='vocabulario.txt')
    lista = list(bag.values())
    print(lista)

    # Caso de teste
    # print(bag)
    # esperado = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1]
    # print(lista == esperado)