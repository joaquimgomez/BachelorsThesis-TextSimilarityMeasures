import argparse
import string
import re
from os import listdir
from os.path import isfile, join
from numpy.random import uniform
from tika import parser as tike_parser
from nltk.tokenize import sent_tokenize
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords

MIN_NUM_SENTENCES_TO_PASS = 0
MAX_NUM_SENTENCES_TO_PASS = 1010
INCREMENT = 10

def PDFtoTXT(path):
	rawFile = tike_parser.from_file(path)
	output = str(rawFile['content'])

	output = output.encode('utf-8', errors='ignore')

	output = output.replace(b'\xc2\xa0', b' ')

	output = output.decode("utf-8")

	#output = output.replace('\n', '')

	return output

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

def passSentences(fromDocument, toDocument, num):
	res = toDocument.copy()

	for i in range(0, num):
		iFrom = int(uniform(0.0, len(fromDocument)))
		iTo = int(uniform(0.0, len(toDocument)))

		res.insert(iTo, fromDocument[iFrom])

	return res

def constructExperimentFilesAndSave(documents, dest, fro, to):
	print("\n\nGenerating documents with sentences from " + str(fro) + " to " + str(to) + ".")

	for i in range(MIN_NUM_SENTENCES_TO_PASS, MAX_NUM_SENTENCES_TO_PASS, INCREMENT):
		newDoc = passSentences(documents[fro], documents[to], i)

		# Save document
		print("Saving file " + str(fro[:-4]) + "_" + str(to[:-4]) + "_" + str(i) + ".txt")

		with open(dest + "/" + str(fro[:-4]) + "_" + str(to[:-4]) + "_" + str(i) + ".txt", 'w') as f:
			for sentence in newDoc:
				f.write(sentence + '\n')

		print("Document" + str(fro) + "_" + str(to) + "_" + str(i) + ".txt" + " saved.")

def main(org, dest, fro, to):
	files = [file for file in listdir(org) if isfile(join(org, file)) and not file.startswith('.')]

	documents = {} # preprocessed documents
	for f in files:
		# PDF to TXT
		text = PDFtoTXT(join(org, f))

		with open(dest + "/" + f[:-4] + ".txt", 'w') as newTXTFile:
			newTXTFile.write(str(text))

		# DOCUMENT PREPROCESSING
		preprocessedDocument = documentPreprocessing(text)

		# DOCUMENT TOKENIZATION
		documents[f] = sent_tokenize(preprocessedDocument)

	# GENERATE EXPERIMENT FILES
	constructExperimentFilesAndSave(documents, dest, fro, to)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Creates the dataset from files in the input directory.")
	parser.add_argument("--origen", "-o",
						help = "Directory of folders containing files.",
						default = "./")
	parser.add_argument("--destination", "-d",
						help = "Directory where dataset goes. The destination folder must not exist.",
						default = "./")
	parser.add_argument("--fromm", "-f",
						help = "Origin of the senteces.")
	parser.add_argument("--to", "-t",
						help = "Destination of the senteces.",
						default = "./")

	args = parser.parse_args()
	main(args.origen, args.destination, args.fromm, args.to)
