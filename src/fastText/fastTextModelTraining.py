import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
import smart_open
import multiprocessing
from os import listdir
from os.path import join, basename, normpath
from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize

# To avoid confusion, the Gensim’s fastText tutorial says that you need to pass a list of tokenized sentences as the input to fastText
class Corpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.sentences = 0

    def __iter__(self):
        for file in listdir(self.dirname):
            with open(join(self.dirname, file)) as file_input:
                corpus = file_input.read()

            raw_sentences = sent_tokenize(corpus)

            for sentence in raw_sentences:
                if len(sentence) > 0:
                    self.sentences += 1
                    yield simple_preprocess(sentence)

		#print(str(self.sentences) + " sentences read.")
		#print(str(len(os.listdir(self.dirname))) + " files read.")

def main(corpusOrg, dest, size, window, min_count, type):
	# Obtain and process the corpus
	print("Preparing corpus...")
	corpus = Corpus(corpusOrg)
	print("Corpus prepared.")

	if type == "PV-DBOW":
		sg = 0
	else:
		sg = 1

	print("Training model...")
	model = FastText(size = int(size),
					window = int(window),
					min_count = int(min_count),
					workers = multiprocessing.cpu_count(),
					sg = sg)

	model.build_vocab(corpus)

	model.train(corpus, total_examples = model.corpus_count, epochs = 10)
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
						help = "Vectors size.",
						default = "300")
	parser.add_argument("--window", "-w",
						help = "Window size.",
						default = "PV-DBOW")
	parser.add_argument("--min", "-m",
						help = "Min count per word.",
						default = "1")
	parser.add_argument("--type", "-t",
						help = "Type of doc2vec training algorithm. It can be 'PV-DM' or 'PV-DBOW'",
						default = "PV-DBOW")

	args = parser.parse_args()
	main(args.corpus, args.destination, args.size, args.window, args.min, args.type)
