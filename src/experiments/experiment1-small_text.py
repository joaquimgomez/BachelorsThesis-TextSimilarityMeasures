#!/usr/bin/env python
# coding: utf-8

import os
import sys
#module_path = os.path.abspath(os.path.join('..'))
#if module_path not in sys.path:
#    sys.path.append(module_path)

import src.similarities.similarities as sim  # Change path to similarity package
from os import listdir
from statistics import mean, stdev
import matplotlib.pyplot as plt
import time
import argparse

def cs(model, experimentsClass, resultsPath, orgExp, docsExp):
    results = []
    times = []

    for doc in docsExp:
        start_time = time.time()
        results.append(experimentsClass.distance(orgExp, doc))
        times.append(time.time() - start_time)

    plt.xlabel('# sentences passed')
    plt.ylabel('distance')
    plt.plot(list(range(0, 110, 10)), results)
    plt.savefig(resultsPath + '/' + model + '_exp1_smallText_Cosine.png')
    plt.close()

    print("Results SCS:")
    for e in results:
        print(e)

    print("Times SCS:")
    for e in times:
        print(e)
    print("Mean execution time: " + str(mean(times)))
    print("Standar deviation execution time: " + str(stdev(times)))

    print("---------------------------")

def scs(model, experimentsClass, resultsPath, orgExp, docsExp):
    results = []
    times = []

    for doc in docsExp:
        start_time = time.time()
        results.append(experimentsClass.distance(orgExp, doc, 0, -1))
        times.append(time.time() - start_time)

    results = [1 - x for x in results]

    plt.xlabel('# sentences passed')
    plt.ylabel('distance')
    plt.plot(list(range(0, 110, 10)), results)
    plt.savefig(resultsPath + '/' + model + '_exp1_smallText_SoftCosine.png')
    plt.close()

    print("Results SCS:")
    for e in results:
        print(e)

    print("Times SCS:")
    for e in times:
        print(e)
    print("Mean execution time: " + str(mean(times)))
    print("Standar deviation execution time: " + str(stdev(times)))

    print("---------------------------")


def wmd(model, experimentsClass, resultsPath, orgExp, docsExp):
    results = []
    times = []

    for doc in docsExp:
        start_time = time.time()
        results.append(experimentsClass.distance(orgExp, doc, 1, False))
        times.append(time.time() - start_time)

    plt.xlabel('# sentences passed')
    plt.ylabel('distance')
    plt.plot(list(range(0, 110, 10)), results)
    plt.savefig(resultsPath + '/' + model + '_exp1_smallText_NormalWordMovers.png')
    plt.close()

    print("Results WMD:")
    for e in results:
        print(e)

    print("Times WMD:")
    for e in times:
        print(e)
    print("Mean execution time: " + str(mean(times)))
    print("Standar deviation execution time: " + str(stdev(times)))

    print("---------------------------")


def rwmd(model, experimentsClass, resultsPath, orgExp, docsExp):
    results = []
    times = []

    for doc in docsExp:
        start_time = time.time()
        results.append(experimentsClass.distance(orgExp, doc, 1, True))
        times.append(time.time() - start_time)

    plt.xlabel('# sentences passed')
    plt.ylabel('distance')
    plt.plot(list(range(0, 110, 10)), results)
    plt.savefig(resultsPath + '/' + model + '_exp1_smallText_RelaxedWordMovers.png')
    plt.close()

    print("Results RWMD:")
    for e in results:
        print(e)

    print("Times RWMD:")
    for e in times:
        print(e)
    print("Mean execution time: " + str(mean(times)))
    print("Standar deviation execution time: " + str(stdev(times)))

    print("---------------------------")

def nrc(model, experimentsClass, resultsPath, orgExp, docsExp):
    results = []
    times = []

    for doc in docsExp:
        start_time = time.time()
        results.append(experimentsClass.distance(orgExp, doc))
        times.append(time.time() - start_time)

    plt.xlabel('# sentences passed')
    plt.ylabel('distance')
    plt.plot(list(range(0, 110, 10)), results)
    plt.savefig(resultsPath + '/' + model + '_exp1_smallText.png')
    plt.close()

    print("Results NRC:")
    for e in results:
        print(e)

    print("Times NRC:")
    for e in times:
        print(e)
    print("Mean execution time: " + str(mean(times)))
    print("Stdev execution time: " + str(stdev(times)))

    print("---------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Performs the Experiment 1.")
    parser.add_argument("--technique", "-t",
                        help = "Technique name",
                        default = "NONE")
    parser.add_argument("--model", "-m",
                        help = "Model path",
                        default = "NONE")
    parser.add_argument("--spacy", "-s",
                        help = "Spacy model path",
                        default = "NONE")
    parser.add_argument("--data", "-d",
                        help = "Experiment data path",
                        default = "NONE")
    parser.add_argument("--results", "-r",
                        help = "Path for the results",
                        default = "NONE")
    args = parser.parse_args()

    # Definitions
    model = args.technique
    modelPath = args.model
    spacyPath = args.spacy
    pathDataExperiment = args.data
    resultsPath = args.results

    if (model != "NONE"):
        # -------- LOAD EXPERIMENT DATA --------

        print("Loading data for the experiment from " + pathDataExperiment + "...")

        print("DOCUMENTS EXPERIMENT 1\n")

        docsExp = []
        orgExp = ""
        for doc in listdir(pathDataExperiment):
            if doc.endswith(".txt"):
                if doc[-7:] == "ORG.txt":
                    orgExp = pathDataExperiment + "/" + doc
                else:
                    docsExp.append(pathDataExperiment + "/" + doc)

        docsExp.sort(key = lambda x: int(x.split("_")[-1][:-4]))

        print("DOCS: " + str(docsExp) + "\n")
        print("ORG: " + orgExp)

        print("Data loaded.")

        # -------- EXPERIMENT --------

        if model == "word2vec":
            print("Experimenting with word2vec.")
            experimentsClass = sim.Word2VecSimilarity(modelPath, spacyPath)

            scs(model, experimentsClass, resultsPath, orgExp, docsExp)

            wmd(model, experimentsClass, resultsPath, orgExp, docsExp)

            rwmd(model, experimentsClass, resultsPath, orgExp, docsExp)

        elif model == "fastText":
            print("Experimenting with fastText.")
            experimentsClass = sim.FastTextSimilarity(modelPath, spacyPath)

            scs(model, experimentsClass, resultsPath, orgExp, docsExp)

            wmd(model, experimentsClass, resultsPath, orgExp, docsExp)

            rwmd(model, experimentsClass, resultsPath, orgExp, docsExp)

        elif model == "GloVe":
            print("Experimenting with GloVe.")
            experimentsClass = sim.GloVeSimilarity(modelPath, spacyPath)

            scs(model, experimentsClass, resultsPath, orgExp, docsExp)

            wmd(model, experimentsClass, resultsPath, orgExp, docsExp)

            rwmd(model, experimentsClass, resultsPath, orgExp, docsExp)

        elif model == "doc2vec":
            print("Experimenting with doc2vec.")
            experimentsClass = sim.Doc2VecSimilarity(modelPath)

            cs(model, experimentsClass, resultsPath, orgExp, docsExp)

        elif model == "ELMo":
            print("Experimenting with ELMo.")
            experimentsClass = sim.ELMoSimilarity(modelPath)

            cs(model, experimentsClass, resultsPath, orgExp, docsExp)

            experimentsClass.closeTFSession()

        elif model == "NRC":
            print("Experimenting with NRC.")
            experimentsClass = sim.NRCSimilarity(modelPath)

            nrc(model, experimentsClass, resultsPath, orgExp, docsExp)
