#!/usr/bin/env python
# coding: utf-8

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram as dendro
import src.similarities.similarities as sim
import numpy as np

class Clustering():
    def computeDistanceMatrix(self, measure, relaxed):
        print("Creating condensed distance matrix...")

        n = len(self.documents)
        m = n
        mat = np.zeros((n, m))

        for i in range(0, n):
            for j in range(0, i + 1):
                print("Computing (" + str(i) + ", " + str(j) + ") of " + "(" + str(n) + ", " + str(m) + ")...")
                if (relaxed != -1):
                    mat[i,j] = self.embedding.distance(self.documents[i], self.documents[j], measure)
                else:
                    mat[i,j] = self.embedding.distance(self.documents[i], self.documents[j], measure, relaxed)

        self.distMat = mat
        print("Condensed distance matrix created.")

        return mat

    def loadDistanceMatrix(self, path):
        self.distMat = np.loadtxt(path)

    def saveDistanceMateix(self, path):
        mat = np.matrix(self.distMat)
        with open(path, 'wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')

    def hierarchicalClustering(self, type):
        self.hierClust = linkage(self.distMat, type)

    def dendrogram(self):
        return dendro(self.hierClust)

    def createEmbeddingObject(self, embedding, embeddingModelPath, embeddingSpacyModelPath):
        if (embedding == "w2v"):
            return sim.Word2VecSimilarity(embeddingModelPath, embeddingSpacyModelPath)
        elif (embedding == "fT"):
            return sim.FastTextSimilarity(embeddingModelPath, embeddingSpacyModelPath)
        elif (embedding == "GV"):
            return sim.GloVeSimilarity(embeddingModelPath, embeddingSpacyModelPath)
        elif (embedding == "d2v"):
            return sim.Doc2VecSimilarity(embeddingModelPath)

    def __init__(self, documents, embedding, embeddingModelPath, embeddingSpacyModelPath):
        print("Creating Clustering object...")
        self.documents = documents
        print("Clustering object created.")

        print("Creating internal objects for the distance computation...")
        self.embedding = self.createEmbeddingObject(embedding, embeddingModelPath, embeddingSpacyModelPath)
        print("Internal objects for the distance computation created.")
