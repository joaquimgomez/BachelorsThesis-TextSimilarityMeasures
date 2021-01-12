#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import sys
import time
import argparse
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import src.similarities.similarities as sim

"""def parDist(documents, embedding, mesure, relaxed, elem):
    print("Computing (" + str(elem[0]) + ", " + str(elem[1]) + ") ...")

    if (measure == 2):
        return embedding.distance(documents[elem[0]], documents[elem[1]])
    else:
        return embedding.distance(documents[elem[0]], documents[elem[1]], measure, relaxed)

def parComputeDistanceMatrix(documents, embedding, measure, relaxed):
    print("Creating condensed distance matrix...")

    n = len(documents)
    m = n
    #mat = np.zeros((n, m))

    lower_triangle = [(i, j) for i in range(0, n) for j in range(0, i + 1)]

    with Pool(cpu_count() - 1) as pool:
        funcAndarg = partial(parDist, documents=documents, embedding=embedding, measure=measure, relaxed=relaxed)
        results = pool.map(funcAndarg, lower_triangle)

        print(results)
        with open("./result.pkl", 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)"""

def computeDistanceMatrix(documents, embedding, measure, relaxed):
    print("Creating condensed distance matrix...")

    n = len(documents)
    m = n
    mat = np.zeros((n, m))

    for i in range(0, n):
        for j in range(0, i + 1):
            print("Computing (" + str(i) + ", " + str(j) + ") of " + "(" + str(n) + ", " + str(m) + ")...")
            if (measure == 2):
                mat[i,j] = embedding.distance(documents[i], documents[j])
            else:
                mat[i,j] = embedding.distance(documents[i], documents[j], measure, relaxed)

    print("Condensed distance matrix created.")

    return mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Performs the distance matrix computation.")
    parser.add_argument("--documents", "-d",
                        help = "Documents path",
                        default = "NONE")
    parser.add_argument("--embedding", "-e",
                        help = "Embedding type",
                        default = "NONE")
    parser.add_argument("--modelembedding", "-me",
                        help = "Embedding model path",
                        default = "NONE")
    parser.add_argument("--scipy", "-s",
                        help = "Embedding Spacy model path",
                        default = "NONE")
    parser.add_argument("--measure", "-m",
                        help = "Measure to be used",
                        default = "NONE")
    parser.add_argument("--relaxed", "-r",
                        help = "Relaxed measure",
                        default = "NONE")
    parser.add_argument("--path", "-p",
                        help = "Distance matrix destination path",
                        default = "NONE")
    args = parser.parse_args()

    documents = args.documents
    embedding = args.embedding
    model = args.modelembedding
    scipy = args.scipy
    measure = int(args.measure)
    relaxed = int(args.relaxed)
    destination = args.path

    print("Retrieving documents...")
    docs = []
    for doc in os.listdir(documents):
        if doc.endswith(".txt"):
            docs.append(documents + doc)
    docs.sort(key = lambda x: int(x.split("/")[-1].split("-")[0]))
    print(docs)
    print("Documents retrieved.")

    print("Creating internal objects for the distance computation...")
    if (embedding == "w2v"):
        m = sim.Word2VecSimilarity(model, scipy)
    elif (embedding == "fT"):
        m = sim.FastTextSimilarity(model, scipy)
    elif (embedding == "GV"):
        m = sim.GloVeSimilarity(model, scipy)
    elif (embedding == "d2v"):
        m = sim.Doc2VecSimilarity(model)
    elif (embedding == "ELMo"):
        m = sim.ELMoSimilarity(model)
    elif (embedding == "NRC"):
        m = sim.NRCSimilarity(model)
    print("Internal objects for the distance computation created.")

    start = time.time()
    #matrix = parComputeDistanceMatrix(docs, m, measure, relaxed) #-- VERSIÓN PARALELA
    matrix = computeDistanceMatrix(docs, m, measure, relaxed) #-- VERSIÓN NO PARALELA
    end = time.time()
    print("The computation taken " + str(end - start) + ".")

    print("Saving the computed distance matrix...")
    npMat = np.matrix(matrix)
    with open(destination, 'wb') as f:
        for line in npMat:
            np.savetxt(f, line, fmt='%.2f')
    print("Distance patrix saved.")

    print("\n\n")
    print("The resulted matrix is the following one:")
    print(npMat)
