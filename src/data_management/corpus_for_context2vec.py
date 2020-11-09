import argparse
from os import listdir, mkdir
from os.path import isfile, join
from nltk import sent_tokenize

def main(org, dest):
	with open(org, 'r') as o:
		corpus = o.read()
		sent_corpus = sent_tokenize(corpus)
		with open(dest, 'w') as d:
			for sent in sent_corpus:
				d.write(sent + '\n')


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
