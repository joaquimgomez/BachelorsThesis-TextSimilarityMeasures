import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from similarities.similarities import NRCSimilarity
from os import listdir
from statistics import mean, stdev
import matplotlib.pyplot as plt
import time

# DEFINITIONS

pathDataExperiment1 = "../../data/experiment1-small_text-continuous/"
pathDataExperiment2 = "../../data/experiment2-big_text-continuous/docs/"

experimentsClass = NRCSimilarity("./NRC/CondComp")

# DOCUMENTS EXPERIMENT 1

print("DOCUMENTS EXPERIMENT 1\n")

docsExp1 = []
orgExp1 = ""
for doc in listdir(pathDataExperiment1):
    if doc.endswith(".txt"):
        if doc[-7:] == "ORG.txt":
            orgExp1 = pathDataExperiment1 + doc
        else:
            docsExp1.append(pathDataExperiment1 + doc)

docsExp1.sort(key = lambda x: int(x.split("_")[-1][:-4]))

print("DOCS: " + str(docsExp1) + "\n")
print("ORG: " + orgExp1)


# DOCUMENTS EXPERIMENT 2

print("\n DOCUMENTS EXPERIMENT 2\n")

docsExp2 = []
orgExp2 = ""
for doc in listdir(pathDataExperiment2):
    if doc.endswith(".txt"):
        if doc == "AE.txt":
            orgExp2 = pathDataExperiment2 + doc
        elif doc == "AI.txt":
            continue
        else:
            docsExp2.append(pathDataExperiment2 + doc)

docsExp2.sort(key = lambda x: int(x.split("_")[-1][:-4]))

print("DOCS: " + str(docsExp2) + "\n")
print("ORG: " + orgExp2)


######### Experiment I (Small texts) - Model validation (Normalized Relative Conditional)

print("EXPERIMENT 1")

results1 = []
times1 = []

for doc in docsExp1:
    start_time = time.time()
    results1.append(experimentsClass.distance(orgExp1, doc))
    times1.append(time.time() - start_time)

plt.xlabel('# sentences passed')
plt.ylabel('distance')

plt.plot(list(range(0, 110, 10)), results1)

plt.savefig('../../figures/NRC_exp1_smallText.png')

plt.close()

print("Results:")
print(results1)

print("Mean execution time: " + str(mean(times1)))
print("Stdev execution time: " + str(stdev(times1)))

print("Times:")
print(times1)

print("\n\n\n")

######### Experiment II (Big texts) - Model validation (Normalized Relative Conditional)

print("EXPERIMENT 2")

results2 = []
times2 = []

for doc in docsExp2:
    start_time = time.time()
    results2.append(experimentsClass.distance(orgExp2, doc))
    times2.append(time.time() - start_time)

plt.xlabel('# sentences passed')
plt.ylabel('distance')

plt.plot(list(range(0, 1010, 10)), results2)

plt.savefig('../../figures/NRC_exp2_BigText.png')

print("Results:")
print(results2)

print("Mean execution time: " + str(mean(times2)))
print("Stdev execution time: " + str(stdev(times2)))

print("Times:")
print(times2)

print("\n\n\n")
