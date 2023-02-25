
import glob
import math
import re
import sys
from collections import defaultdict
from functools import reduce
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
CORPUS = "Corpus/*"
file_name = dict()
N = 0
vocabulary = set()
postings = defaultdict(dict)
df = defaultdict(int)
length = defaultdict(float)


def main():
    get_corpus()
    normalization()
    ini_df()
    ini_len()
    while True:
        scores = do_search()
        for (id, score) in scores:
         if score != 0.0:
            print((str(score)[:5], file_name[id]))


def get_corpus():
    global file_name, N
    documents = glob.glob(CORPUS)
    N = len(documents)
    file_name = dict(zip(range(N), documents))
    print(N)
    print(file_name)

def remove_special_characters(text):
    regex = re.compile(r"[^a-zA-Z0-9\s]")
    return re.sub(regex, "", text)


def remove_digits(text):
    regex = re.compile(r"\d")
    return re.sub(regex, "", text)

def normalization():
    global vocabulary, postings
    for id in file_name:
        with open(file_name[id], "r") as f:
            document = f.read()
        document = remove_special_characters(document)
        document = remove_digits(document)
        terms = tokenize(document)
        unique_terms = set(terms)
        vocabulary = vocabulary.union(unique_terms)
        for term in unique_terms:
            postings[term][id] = terms.count(term)


def tokenize(document):
    terms = word_tokenize(document)
    terms = [term.lower() for term in terms if term not in STOPWORDS]
    return terms


def ini_df():
    global df
    for term in vocabulary:
        df[term] = len(postings[term])


def ini_len():
    global length
    for id in file_name:
        l = 0
        for term in vocabulary:
            l += tf(term, id) 
        length[id] = math.sqrt(l)


def tf(term, id):
    if id in postings[term]:
        return  (1+math.log(postings[term][id]))

    else:
        return 0.0


def idf(term):
    if term in vocabulary:
        return math.log(N / df[term],2)
    else:
        return 0.0
    
def do_search():
    query = tokenize(input("Quesy:"))

    # Exit if query is empty
    if query == []:
        sys.exit()

    scores = sorted(
        [(id, simi(query, id)) for id in range(N)],
        key=lambda x: x[1],
        reverse=True,
    )

    return scores


def intersection(sets):
    return reduce(set.intersection, [s for s in sets])


def simi(query, id):
    simi = 0.0

    for term in query:

        if term in vocabulary:
            simi += math.sqrt(tf(term, id) * idf(term))

    simi = simi / length[id]

    return simi


if __name__ == "__main__":
    main()
