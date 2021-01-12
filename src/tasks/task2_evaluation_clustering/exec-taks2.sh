#!/usr/bin/env bash

# ------------- VARIABLES -------------

dataPath="./data/preprocessed_clustering/"

# Change path to models source
word2vecModel="./models/word2vec/word2vec.model"
fastTextModel="./models/fastText/fastText.model"
GloVeModel="./models/GloVe/GloVe.txt"
doc2vecModel="./models/doc2vec/doc2vec.model"
ELMoModel="./models/ELMo/"
NRCModel="./src/Normalized_Relative_Compression/NRC/CondComp"

word2vecModelSpaCy="./models/word2vec/spacy.word2vec.model"
fastTextModelSpaCy="./models/fastText/spacy.fastText.model"
GloVeModelSpaCy="./Experimentacion/models/GloVe/spacy.GloVe.model"


# ------------- EXPERIMENTS -------------

#conda activate tfg

# -------------- w2v --------------
echo "w2v"
# -------- SCS --------
python3 distanceMatrixCalculation.py -d $dataPath -e w2v --modelembedding $word2vecModel -s $word2vecModelSpaCy --measure 0 -r 2 -p ./results/distance_matrix_w2v_scs.txt
# -------- WMD --------
python3 distanceMatrixCalculation.py -d $dataPath -e w2v --modelembedding $word2vecModel -s $word2vecModelSpaCy --measure 1 -r 0 -p ./new-results/distance_matrix_w2v_wmd.txt
# -------- RWMD -------
python3 distanceMatrixCalculation.py -d $dataPath -e w2v --modelembedding $word2vecModel -s $word2vecModelSpaCy --measure 1 -r 1 -p ./results/distance_matrix_w2v_rwmd.txt


# -------------- fT --------------
echo "\n\n"
echo "fT"
# -------- SCS --------
python3 distanceMatrixCalculation.py -d $dataPath -e fT --modelembedding $fastTextModel -s $fastTextModelSpaCy --measure 0 -r 2 -p ./results/distance_matrix_fT_scs.txt
# -------- WMD --------
python3 distanceMatrixCalculation.py -d $dataPath -e fT --modelembedding $fastTextModel -s $fastTextModelSpaCy --measure 1 -r 0 -p ./new-results/distance_matrix_fT_wmd.txt
# -------- RWMD -------
python3 distanceMatrixCalculation.py -d $dataPath -e fT --modelembedding $fastTextModel -s $fastTextModelSpaCy --measure 1 -r 1 -p ./results/distance_matrix_fT_rwmd.txt


# -------------- GV --------------
echo "\n\n"
echo "GV"
# -------- SCS --------
python3 distanceMatrixCalculation.py -d $dataPath -e GV --modelembedding $GloVeModel -s $GloVeModelSpaCy --measure 0 -r 2 -p ./results/distance_matrix_GV_scs.txt
# -------- WMD --------
python3 distanceMatrixCalculation.py -d $dataPath -e GV --modelembedding $GloVeModel -s $GloVeModelSpaCy --measure 1 -r 0 -p ./new-results/distance_matrix_GV_wmd.txt
# -------- RWMD -------
python3 distanceMatrixCalculation.py -d $dataPath -e GV --modelembedding $GloVeModel -s $GloVeModelSpaCy --measure 1 -r 1 -p ./results/distance_matrix_GV_rwmd.txt


# -------------- d2v --------------
echo "\n\n"
echo "d2v"
# -------- CS ---------
python3 distanceMatrixCalculation.py -d $dataPath -e d2v --modelembedding $doc2vecModel -s 2 --measure 2 -r 2 -p ./results/distance_matrix_d2v_cs.txt


# -------------- ELMo --------------
echo "\n\n"
echo "ELMo"
# -------- CS ---------
python3 distanceMatrixCalculation.py -d $dataPath -e ELMo --modelembedding $ELMoModel -s 2 --measure 2 -r 2 -p ./new-results/distance_matrix_ELMo_cs.txt


# -------------- NRC --------------
echo "\n\n"
echo "NRC"
python3 distanceMatrixCalculation.py -d $dataPath -e NRC --modelembedding $NRCModel -s 2 --measure 2 -r 2 -p  ./results/distance_matrix_NRC.txt
