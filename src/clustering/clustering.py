#!/usr/bin/env python
# coding: utf-8

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from similarities import similarities as sim
from scipy.cluster.hierarchy import ward, fcluster,
from scipy.cluster.hierarchy import dendrogram as dendro

class Clustering():
    def distanceMatrix(self, embedding, measure, relaxed):
        n, m = len(self.documents)
        mat = np.zeros((n, m))

        for i in range(0, n):
            for j in range(0, m):
                if (relaxed != -1):
                    mat[i,j] = self.embedding.distance(self.documents[i], self.documents[j])
                else:
                    mat[i,j] = self.embedding.distance(self.documents[i], self.documents[j], relaxed)

        return mat

    def hierarchicalClustering(self):
        self.hierClust = ward(self.distMat)

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

    def __init__(self, documents, embedding, embeddingModelPath, measure, relaxed):
        print("Creating Clustering object...")
        self.documents = documents
        print("Clustering object created.")

        print("Creating internal objects for the distance computation...")
        self.embedding = createEmbeddingObject(embedding, embeddingModelPath, embeddingSpacyModelPath)
        print("Internal objects for the distance computation created.")

        print("Creating condensed distance matrix...")
        self.distMat = distanceMatrix(embedding, measure, relaxed)
        print("condensed distance matrix created")
