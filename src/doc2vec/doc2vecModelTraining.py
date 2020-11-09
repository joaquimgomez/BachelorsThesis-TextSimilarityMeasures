import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from os import listdir
from os.path import join, splitext, basename, normpath
import argparse
import smart_open
import multiprocessing

def getCorpus(org):
	count = 0

	for file in listdir(org):
		if file.endswith(".txt"):
			with smart_open.open(join(org, file), 'r', encoding='utf-8') as f:
				fileitself = simple_preprocess(f.read())
				name = splitext(file)[0]
				yield TaggedDocument(fileitself, [name])

				count = count + 1

	print(str(count) + " files read.")


def main(corpusOrg, dest, size, type):
	print("Reading corpus...")
	corpus = list(getCorpus(corpusOrg))
	print("Corpus read.")

	t = 0
	if type == "PV-DM":
		t = 1

	model = Doc2Vec(vector_size = size, epochs = 10, dm = t, workers = multiprocessing.cpu_count())

	print("Building model...")
	model.build_vocab(corpus)
	print("Model builded.")

	print("Training model...")
	model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
	print("Model trained.")

	print("Savind model...")
	dest = join(dest, basename(normpath(dest)) + '.model')
	model.save(dest)
	print("Model saved.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Generates a doc2vec model for a given set of files.")
	parser.add_argument("--corpus", "-c",
						help = "Directory where the files are.",
						default = "./")
	parser.add_argument("--destination", "-d",
						help = "Directory where the model goes.",
						default = "./")
	parser.add_argument("--size", "-s",
						help = "Vector size.",
						default = "300")
	parser.add_argument("--type", "-t",
						help = "Type of doc2vec training algorithm. It can be 'PV-DM' or 'PV-DBOW'",
						default = "PV-DBOW")

	args = parser.parse_args()
	main(args.corpus, args.destination, int(args.size), args.type)
