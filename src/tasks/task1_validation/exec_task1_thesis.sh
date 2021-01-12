#!/usr/bin/env bash

# ------------- VARIABLES -------------

basePathExp2BigTextDest="/Users/joaquingomez/Downloads/Experimentacion/experiments/Experiment2-BigText" # Change path to experiments results destination
dataExp2BigTextDest="$basePathExp2BigTextDest/"


basePathExp2BigTextOrg="/Users/joaquingomez/Downloads/Experimentacion/data/experiment2-big_text/docs" # Change path to experiments data sources
dataExp2BigTextOrg="$basePathExp2BigTextOrg/"


 # Change path to models source
word2vecModel="/Users/joaquingomez/Downloads/Experimentacion/models/word2vec/word2vec.model"
fastTextModel="/Users/joaquingomez/Downloads/new_fT/fT/fT.model"
GloVeModel="/Users/joaquingomez/Downloads/Experimentacion/models/GloVe/GloVe.txt"
doc2vecModel="/Users/joaquingomez/Downloads/Experimentacion/models/doc2vec/doc2vec.model"
ELMoModel="/Users/joaquingomez/Downloads/Experimentacion/models/ELMo/"
NRCModel="/Users/joaquingomez/Downloads/Experimentacion/src/Normalized_Relative_Compression/NRC/CondComp"

word2vecModelSpaCy="/Users/joaquingomez/Downloads/Experimentacion/models/word2vec/spacy.word2vec.model"
fastTextModelSpaCy="/Users/joaquingomez/Downloads/Experimentacion/models/fastText/spacy.fastText.model"
GloVeModelSpaCy="/Users/joaquingomez/Downloads/Experimentacion/models/GloVe/spacy.GloVe.model"


# ------------- EXPERIMENTS -------------

#conda activate tfg

# ----- word2vec -----

echo "word2vec running"
python3 experiment2-big_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $dataExp2BigTextOrg -r $dataExp2BigTextDest > $dataExp2BigTextDest/results_word2vec.txt
echo "\n"


# ----- fastText -----

echo "fastText running"
python3 experiment2-big_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $dataExp2BigTextOrg -r $dataExp2BigTextDest #> $dataExp2BigTextDest/results_fastText.txt
echo "\n"


# ----- GloVe -----

echo "GloVe running"
python3 experiment2-big_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $dataExp2BigTextOrg -r $dataExp2BigTextDest > $dataExp2BigTextDest/results_GloVe.txt
echo "\n"


# ----- doc2vec -----

echo "doc2vec running"
python3 experiment2-big_text.py -t doc2vec -m $doc2vecModel -s NONE -d $dataExp2BigTextOrg -r $dataExp2BigTextDest > $dataExp2BigTextDest/results_doc2vec.txt
echo "\n"


# ----- ELMO -----

echo "ELMo running"
python3 experiment2-big_text.py -t ELMo -m $ELMoModel -s NONE -d $dataExp2BigTextOrg -r $dataExp2BigTextDest > $dataExp2BigTextDest/results_ELMo.txt
echo "\n"


# ----- NRC -----

echo "NRC running"
python3 experiment2-big_text.py -t NRC -m $NRCModel -s NONE -d $dataExp2BigTextOrg -r $dataExp2BigTextDest > $dataExp2BigTextDest/results_NRC.txt
echo "\n"
