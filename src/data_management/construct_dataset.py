import argparse
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, join, exists
from itertools import permutations
from numpy.random import uniform
from nltk.tokenize import sent_tokenize

MIN_NUM_SENTENCES_TO_PASS = 10
MAX_NUM_SENTENCES_TO_PASS = 100
INCREMENT = 10

def passSentences(fromDocument, toDocument, num):
	res = toDocument.copy()

	for i in range(0, num):
		iFrom = int(uniform(0.0, len(fromDocument)))
		iTo = int(uniform(0.0, len(toDocument)))

		res.insert(iTo, fromDocument[iFrom])

	return res

def constructAndSaveDataset(documents, dest):
	keys = documents.keys()
	perms = list(permutations(keys, 2))

	print("A dataset with " + str(len(perms)) + " permutations will be generated.")

	mkdir(dest)

	datasetIndex = pd.DataFrame(columns=['id', 'from', 'to', 'number_of_passed_sentences'])

	currentId = 1
	for (fromDocument, toDocument) in perms:
		print("\n\nGenerating documents with sentences from " + str(fromDocument) + " to " + str(toDocument) + ".")

		for i in range(MIN_NUM_SENTENCES_TO_PASS, MAX_NUM_SENTENCES_TO_PASS, INCREMENT):
			newDoc = passSentences(documents[fromDocument], documents[toDocument], i)

			# Save document
			print("Saving file " + str(currentId) + "_" + str(fromDocument) + "_" + str(toDocument) + "_" + str(i) + ".txt")

			datasetIndex = datasetIndex.append({'id': currentId, 'from': fromDocument, 'to': toDocument, 'number_of_passed_sentences': i}, ignore_index=True)

			with open(dest + str(currentId) + "_" + str(fromDocument) + "_" + str(toDocument) + "_" + str(i) + ".txt", 'w') as f:
				for sentence in newDoc:
					f.write(sentence + '\n')

			print("Document" + str(currentId) + "saved.")

			currentId = currentId + 1

	return datasetIndex

def obtainFileContents(files):
	documents = {}

	print("Obtaining content from preprocessed files.")
	for (file, path) in files:
		print("Obtaining content from file " + file + ".")

		with open(path) as f:
			doc = f.read()
			sentences = sent_tokenize(doc)
			documents[file[:-4]] = sentences

	return documents

def main(org, dest):
	# Load paths of preprocessed files
	files = [(file, join(org, file)) for file in listdir(org) if isfile(join(org, file)) and not file.startswith('.')]

	# Obtain content of preprocessed files
	documents = obtainFileContents(files)

	# Create meta folder
	if not exists('./meta/'):
		mkdir('./meta/')

	# Construct and save dataset
	print("Generating and saving dataset (with its index):")

	indexDataset = constructAndSaveDataset(documents, dest)
	indexDataset.to_csv('./meta/dataset_index.csv', index=False)

	print("Dataset generated and saved.")
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
