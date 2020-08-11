import argparse
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join, exists
from itertools import permutations
from numpy.random import uniform

MIN_NUM_SENTENCES_TO_PASS = 1
MAX_NUM_SENTENCES_TO_PASS = 40
INCREMENT = 5

def passSentences(fromDocument, toDocument, num):
	res = toDocument.copy()

	for i in range(0, num):
		iFrom = int(uniform(0.0, len(fromDocument)))
		iTo = int(uniform(0.0, len(toDocument)))

		res.insert(iTo, fromDocument[iFrom])

	return res

def constructDataset(documents):
	dataset = {}

	keys = documents.keys()
	perms = list(permutations(keys, 2))

	print("A dataset with " + str(len(perms)) + " permutations will be generated.")

	currentId = 1
	for (fromDocument, toDocument) in perms:
		print("Generating documents with sentences from " + str(fromDocument) + " to " + str(toDocument) + ".")

		for i in range(MIN_NUM_SENTENCES_TO_PASS, MAX_NUM_SENTENCES_TO_PASS, INCREMENT):
			dataset[(currentId, fromDocument, toDocument, i)] = passSentences(documents[fromDocument], documents[toDocument], i)
			currentId = currentId + 1

	return dataset

def saveDataset(dataset, dest):
	datasetIndex = pd.DataFrame(columns=['id', 'from', 'to', 'number_of_passed_sentences'])

	mkdir(dest)

	for (id, fromDocument, toDocument, num) in dataset.keys():
		print("Saving file " + str(id) + "_" + str(fromDocument) + "_" + str(toDocument) + "_" + str(num) + ".txt")

		datasetIndex = datasetIndex.append({'id': id, 'from': fromDocument, 'to': toDocument, 'number_of_passed_sentences': num}, ignore_index=True)

		with open(dest + str(id) + "_" + str(fromDocument) + "_" + str(toDocument) + "_" + str(num) + ".txt", 'w') as f:
			for sentence in dataset[(id, fromDocument, toDocument, num)]:
				f.write(sentence + '\n')

	return datasetIndex

def obtainFileContents(files):
	documents = {}

	print("Obtaining content from preprocessed files.")
	for (file, path) in files:
		print("Obtaining content from file " + file + ".")

		f = open(path)
		fLines = f.readlines()

		documents[file[:-4]] = fLines

		f.close()

	return documents

def main(org, dest):
	# Load paths of preprocessed files
	files = [(file, join(org, file)) for file in listdir(org) if isfile(join(org, file)) and not file.startswith('.')]

	# Obtain content of preprocessed files
	documents = obtainFileContents(files)

	# Constructing dataset
	print("Constructing dataset:")
	dataset = constructDataset(documents)

	print("Dataset with " + str(len(dataset.keys())) + " documents constructed.")
	print("\n\n")

	# Create meta folder
	if not exists('./meta/'):
		mkdir('./meta/')

	# Generate index of dataset and save dataset
	print("Generating dataset and savind index of dataset:")
	indexDataset = saveDataset(dataset, dest)
	indexDataset.to_csv('./meta/dataset_index.csv', index=False)

	print("Dataset saved.")
	print("Index generated and saved.")
	print("\n\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Creates the dataset from files in the input directory.")
	parser.add_argument("--origen", "-o",
						help = "Directory of folders containing preprocessed files.",
						default = "./")
	parser.add_argument("--destination", "-d",
						help = "Directory where dataset goes. The destination folder must not exist.",
						default = "./dataset/")

	args = parser.parse_args()
	main(args.origen, args.destination)
