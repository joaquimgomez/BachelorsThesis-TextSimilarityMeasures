from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
from gensim import corpora
from gensim.matutils import softcossim
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile
from nltk.tokenize import sent_tokenize
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import doc2vec
from scipy.spatial.distance import cosine
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
import tensorflow_hub as hub
import tensorflow as tf
import subprocess
import spacy
import wmd
import threading

class Word2VecAndFastTextSimilarity:
	def softCosineSimilarity(self, doc1, doc2):
		doc1vec = doc1.split()
		doc2vec = doc2.split()

		docs = [doc1vec, doc2vec]
		dictionary = corpora.Dictionary(docs)

		similarity_matrix = self.model.wv.similarity_matrix(dictionary)

		bow1 = dictionary.doc2bow(doc1vec)
		bow2 = dictionary.doc2bow(doc2vec)

		return softcossim(bow1, bow2, similarity_matrix)

	def wordMoversDistance(self, doc1, doc2, relaxed):
		if relaxed:
			# Relaxed WMD
			return (self.spacyModel(doc1)).similarity(self.spacyModel(doc2))
		else:
			doc1_ll = [simple_preprocess(sent) for sent in sent_tokenize(doc1)]
			doc1_ll = [item for sublist in doc1_ll for item in sublist]

			doc2_ll = [simple_preprocess(sent) for sent in sent_tokenize(doc2)]
			doc2_ll = [item for sublist in doc2_ll for item in sublist]

			return self.model.wmdistance(doc1_ll, doc2_ll)

	def distance(self, pathDoc1, pathDoc2, typeDistance, relaxed):
		doc1 = open(pathDoc1, "r")
		contentDoc1 = doc1.read()
		doc1.close()

		doc2 = open(pathDoc2, "r")
		contentDoc2 = doc2.read()
		doc1.close()

		if typeDistance == 0:
			return self.softCosineSimilarity(contentDoc1, contentDoc2)
		else:
			return self.wordMoversDistance(contentDoc1, contentDoc2, relaxed)

	def __init__(self, model, spacyModel):
		self.model = model
		self.spacyModel = spacyModel
		self.spacyModel.add_pipe(wmd.WMD.SpacySimilarityHook(self.spacyModel), last = True)


class Word2VecSimilarity(Word2VecAndFastTextSimilarity):
	def distance(self, pathDoc1, pathDoc2, typeDistance, relaxed):
		return super().distance(pathDoc1, pathDoc2, typeDistance, relaxed)

	def __init__(self, modelPath, spacyPath):
		self.path = modelPath
		self.spacyPath = spacyPath

		model = Word2Vec.load(self.path)
		spacyModel = spacy.load(self.spacyPath)

		# Normalize model (L2-normalize)
		model.init_sims(replace=True)

		# Model to superclass
		super().__init__(model, spacyModel)


class FastTextSimilarity(Word2VecAndFastTextSimilarity):
	def distance(self, pathDoc1, pathDoc2, typeDistance, relaxed):
		return super().distance(pathDoc1, pathDoc2, typeDistance, relaxed)

	def __init__(self, modelPath, spacyPath):
		self.path = modelPath
		self.spacyPath = spacyPath

		model = FastText.load(self.path)
		spacyModel = spacy.load(self.spacyPath)

		# Normalize model (L2-normalize)
		model.init_sims(replace = True)

		# Model to superclass
		super().__init__(model, spacyModel)


class GloVeSimilarity(Word2VecAndFastTextSimilarity):
	def distance(self, pathDoc1, pathDoc2, typeDistance, relaxed):
		return super().distance(pathDoc1, pathDoc2, typeDistance, relaxed)

	def __init__(self, modelPath, spacyPath):
		self.path = modelPath
		self.spacyPath = spacyPath

		glove_file = datapath(self.path)
		tmp_file = get_tmpfile("word2vec.txt")

		_ = glove2word2vec(glove_file, tmp_file)

		model = KeyedVectors.load_word2vec_format(tmp_file)
		spacyModel = spacy.load(self.spacyPath)

		# Normalize model (L2-normalize)
		model.init_sims(replace = True)

		# Model to superclass
		super().__init__(model, spacyModel)


class Doc2VecSimilarity():
	def cosineSimilarity(vecDoc1, vecDoc2):
		return 1 - cosine(vec1, vec2)

	def distance(self, pathDoc1, pathDoc2):
		f1 = open(pathDoc1, 'r')
		f1_text = f1.read()
		f1.close()

		f2 = open(pathDoc2, 'r')
		f2_text = f2.read()
		f2.close()

		vecDoc1 = self.model.infer_vector(simple_preprocess(f1_text))
		vecDoc2 = self.model.infer_vector(simple_preprocess(f2_text))

		return cosine(vecDoc1, vecDoc2)

	def __init__(self, modelPath):
		self.path = modelPath

		self.model = doc2vec.Doc2Vec.load(self.path)

		# Normalize model (L2-normalize)
		self.model.init_sims(replace = True)

class BERTSimilarity():
	def startBERTServer(self):
		args = get_args_parser().parse_args(['-model_dir', self.path,
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', 'NONE',
                                     '-mask_cls_sep',
                                     '-cpu'])
		self.server = BertServer(args)
		self.server.start()

		self.bertClient = BertClient()

	def killBERTServer(self):
		self.server.shutdown(port=5555)

	def distance(self, pathDoc1, pathDoc2):
		f1 = open(pathDoc1, 'r')
		f1_text = f1.read()
		f1.close()

		f2 = open(pathDoc2, 'r')
		f2_text = f2.read()
		f2.close()

		vectors = self.bertClient.encode([f1_text, f2_text])

		return(cosine(vectors[0], vectors[1]))

	def __init__(self, modelPath):
		self.path = modelPath

class ELMoSimilarity:
	def elmoEmbedding(self, text):
		embedding = self.model(text, signature = "default", as_dict = True)["elmo"]

		# return average of ELMo features
		return self.tfSession.run(tf.reduce_mean(embedding, 1))

	def distance(self, pathDoc1, pathDoc2):
		f1 = open(pathDoc1, 'r')
		f1_text = f1.read()
		f1.close()

		f2 = open(pathDoc2, 'r')
		f2_text = f2.read()
		f2.close()

		embeddingDoc1 = self.elmoEmbedding([f1_text])
		embeddingDoc2 = self.elmoEmbedding([f2_text])

		return(cosine(embeddingDoc1, embeddingDoc2))

	def closeTFSession():
		self.tfSession.close()

	def __init__(self, modelPath):
		self.path = modelPath

		self.model = hub.Module(self.path)

		self.tfSession = tf.Session()
		self.tfSession.run(tf.global_variables_initializer())
		self.tfSession.run(tf.tables_initializer())


class NRCSimilarity:
	def distance(self, reference, target):
		#cmd = "%s -g 0.98 -r %s -tk 5 1/100 -tk 7 1/100 -t %s"%(self.path, reference, target)
		cmd = "%s -rk 2 -rk 3 -r %s -t %s"%(self.path, reference, target)
		cmd = cmd + " | tail -1"

		output = str(subprocess.check_output(cmd.split()))

		resultExecution = output.split(',')[-1][:-4]

		return float(resultExecution)

	def __init__(self, codePath):
		self.path = codePath
