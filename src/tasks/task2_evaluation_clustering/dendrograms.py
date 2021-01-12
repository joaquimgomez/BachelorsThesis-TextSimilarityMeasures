from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

def hierClustering(doc, distanceMatrix, labels, destination):
    # Average clustering
    #clust = linkage(distanceMatrix, 'average', optimal_ordering=True)
    #fig = plt.figure(figsize=(5,10), dpi=300)
    #dn = dendrogram(clust, orientation='right', labels = labels)
    #plt.savefig(destination + "/" + doc[:-4] + '_average.png', format='png', bbox_inches='tight')

    # Weighted clustering
    clust = linkage(distanceMatrix, 'weighted', optimal_ordering=True)
    fig = plt.figure(figsize=(5,10), dpi=300)
    dn = dendrogram(clust, orientation='right', labels = labels)
    plt.savefig(destination + "/" + doc[:-4] + '_weighted.png', format='png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Performs the Experiment 2.")
    parser.add_argument("--origin", "-o",
                        help = "Folder with distance matrixs.",
                        default = "NONE")
    parser.add_argument("--destination", "-d",
                        help = "Destination folder for the dendrograms.",
                        default = "NONE")
    parser.add_argument("--labels", "-l",
                        help = "Path to original documents used for clustering.",
                        default = "NONE")
    args = parser.parse_args()

    org = args.origin
    dest = args.destination
    originalDocuemnts = args.labels

    # Obtain dendrogram labels
    docs = []
    for doc in os.listdir(originalDocuemnts):
        if doc.endswith(".txt"):
            docs.append(originalDocuemnts.split("/")[-1] + doc)

    docs.sort(key = lambda x: int(x.split("/")[-1].split("-")[0]))

    labels = [doc.split("-")[-1][:-4] for doc in docs]

    # Obtain distance matrixs and compute clustering
    for doc in os.listdir(org):
        if doc.endswith(".txt"):
            distanceMatrix = np.loadtxt(org + "/" + doc)
            np.fill_diagonal(distanceMatrix, 0)

            hierClustering(doc, distanceMatrix, labels, dest)
