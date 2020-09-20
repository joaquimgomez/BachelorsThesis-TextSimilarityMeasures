import argparse
import nltk.data
import string
import pandas as pd
import re
from os import listdir, mkdir
from os.path import isfile, join

from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

def saveFiles(documents, dest):
	mkdir(dest)

	for doc in documents:
		with open(dest + str(doc) + '.txt', 'w') as f:
			f.write(documents[doc])
			#for sentence in documents[doc]:
			#	f.write(sentence + '\n')

def documentPreprocessing(document):
	# Filter for non-printable characters
	filter_printable = lambda x: x in string.printable

	# Stemmer
	porter = PorterStemmer()

	#for i in range(0, len(document)):
	doc = document

	# Lowercasing
	doc = doc.lower()

	# Remove emails and web addresses
	doc = re.sub(r'\S*@\S*\s?', '', doc, flags = re.MULTILINE)
	doc = re.sub(r'http\S+', '', doc, flags = re.MULTILINE)

	# Erase non-printable characters
	doc = ''.join(filter(filter_printable, doc))

	# Remove Stopwords (using gensim stopwords set)
	doc = remove_stopwords(doc)

	# Stemming
	doc = porter.stem_sentence(doc)

	return doc

def obtainFileContents(index):
	#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	documents = {}

	print("Obtaining content from indexed files.")
	for ind, row in index.iterrows():
		print("Obtaining content from file " + row['file_name'] + ".")

		f = open(row['file_path'])
		fContent = f.read()

		documents[row['id']] = fContent #tokenizer.tokenize(fContent)

		f.close()

	print("\n\n")

	return documents

def generateIndex(folders):
	print("Constructing index from files:")

	index = pd.DataFrame(columns=['id', 'category', 'file_name', 'file_path'])

	currentId = 1
	for (folderName, path) in folders:
		print("Indexing files from folder " + folderName + ".")

		files = [(file, join(path, file)) for file in listdir(path) if isfile(join(path, file)) and not file.startswith('.')]

		for (file, filePath) in files:
			index = index.append({'id': currentId, 'category': folderName, 'file_name': file, 'file_path': filePath}, ignore_index=True)

			currentId = currentId + 1

	print("\nTotal number of indexed files: " + str(len(index.index)))
	print("Indexed files:")
	print(index)
	print("\n\n")

	return index

def main(org, dest):
	# Obtain all the folders
	folders = [(folder, join(org, folder)) for folder in listdir(org) if not isfile(join(org, folder)) and not folder.startswith('.')]

	# Generate an index for all files
	index = generateIndex(folders)

	# Save index to csv
	mkdir('./meta/')
	index.to_csv('./meta/pdfs_index.csv', index=False)

	# Obtain content of all documents in index
	documents = obtainFileContents(index)

	# Preprocess documents
	print("Preprocessing loaded documents:")
	for doc in documents:
		print("Preprocessing document with id " + str(doc) + ".")
		documents[doc] = documentPreprocessing(documents[doc])

	print("\n\n")

	# Save preprocessed files
	print("Saving preprocessed files.")
	saveFiles(documents, dest)
	print("\n\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Creates the dataset from files in the input directory.")
	parser.add_argument("--origen", "-o",
						help = "Directory of folders containing files.",
						default = "./")
	parser.add_argument("--destination", "-d",
						help = "Directory where dataset goes. The destination folder must not exist.",
						default = "./")

	args = parser.parse_args()
	main(args.origen, args.destination)
