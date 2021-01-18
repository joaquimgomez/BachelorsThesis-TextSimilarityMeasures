import argparse
from nltk.tokenize import word_tokenize
from os import listdir, mkdir
from os.path import isfile, join

def main(org, dest):
	files = [file for file in listdir(org) if isfile(join(org, file)) and not file.startswith('.')]

	docs = []
	for file in files:
		with open(join(org, file), 'r') as f:
			fileContent = f.read()
			tokenization = word_tokenize(fileContent)
			docs.append(tokenization)

	corpus = open(dest + "corpus.txt", 'w')
	for doc in docs:
		string = ' '.join(doc)
		corpus.write(string)
		corpus.write("\n\n\n\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Creates the dataset from files in the input directory.")
	parser.add_argument("--origen", "-o",
						help = "Directory of folders containing preprocessed files.",
						default = "./")
	parser.add_argument("--destination", "-d",
						help = "Directory where the final document for GloVe goes.",
						default = "./")

	args = parser.parse_args()
	main(args.origen, args.destination)
