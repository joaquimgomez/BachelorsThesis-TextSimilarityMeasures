#!/usr/bin/env bash

# ------------- VARIABLES -------------

basePathExp1SmallTextDest="/Users/joaquingomez/Downloads/Experimentacion/experiments/Experiment1-SmallText" # Change path to experiments results destination

dataExp1SmallTextDiffGroupDest="$basePathExp1SmallTextDest/diffGroup"
DIR_180_6_Dest="$dataExp1SmallTextDiffGroupDest/180-CDIF_6-ANT"
DIR_180_168_Dest="$dataExp1SmallTextDiffGroupDest/180-CDIF_167-ADR&M"
DIR_180_579_Dest="$dataExp1SmallTextDiffGroupDest/180-CDIF_579-UMA"
DIR_180_735_Dest="$dataExp1SmallTextDiffGroupDest/180-CDIF_735-AIEM"
DIR_180_836_Dest="$dataExp1SmallTextDiffGroupDest/180-CDIF_836-IDEAI"

dataExp1SmallTextSameGroupDest="$basePathExp1SmallTextDest/sameGroup"
DIR_180_185_Dest="$dataExp1SmallTextSameGroupDest/180-CDIF_185-CDIF"
DIR_180_194_Dest="$dataExp1SmallTextSameGroupDest/180-CDIF_194-CDIF"
DIR_180_206_Dest="$dataExp1SmallTextSameGroupDest/180-CDIF_206-CDIF"
DIR_180_207_Dest="$dataExp1SmallTextSameGroupDest/180-CDIF_207-CDIF"
DIR_180_235_Dest="$dataExp1SmallTextSameGroupDest/180-CDIF_235-CDIF"


basePathExp1SmallTextOrg="/Users/joaquingomez/Downloads/Experimentacion/data/experiment1-small_text" # Change path to experiments data sources

dataExp1SmallTextDiffGroupOrg="$basePathExp1SmallTextOrg/diffGroup"
DIR_180_6_Org="$dataExp1SmallTextDiffGroupOrg/180-CDIF_6-ANT"
DIR_180_168_Org="$dataExp1SmallTextDiffGroupOrg/180-CDIF_167-ADR&M"
DIR_180_579_Org="$dataExp1SmallTextDiffGroupOrg/180-CDIF_579-UMA"
DIR_180_735_Org="$dataExp1SmallTextDiffGroupOrg/180-CDIF_735-AIEM"
DIR_180_836_Org="$dataExp1SmallTextDiffGroupOrg/180-CDIF_836-IDEAI"

dataExp1SmallTextSameGroupOrg="$basePathExp1SmallTextOrg/sameGroup"
DIR_180_185_Org="$dataExp1SmallTextSameGroupOrg/180-CDIF_185-CDIF"
DIR_180_194_Org="$dataExp1SmallTextSameGroupOrg/180-CDIF_194-CDIF"
DIR_180_206_Org="$dataExp1SmallTextSameGroupOrg/180-CDIF_206-CDIF"
DIR_180_207_Org="$dataExp1SmallTextSameGroupOrg/180-CDIF_207-CDIF"
DIR_180_235_Org="$dataExp1SmallTextSameGroupOrg/180-CDIF_235-CDIF"


 # Change path to models source
word2vecModel="/Users/joaquingomez/Downloads/Experimentacion/models/word2vec/word2vec.model"
fastTextModel="/Users/joaquingomez/Downloads/Experimentacion/models/fastText/fastText.model"
GloVeModel="/Users/joaquingomez/Downloads/Experimentacion/models/GloVe/GloVe.txt"
doc2vecModel="/Users/joaquingomez/Downloads/Experimentacion/models/doc2vec/doc2vec.model"

word2vecModelSpaCy="/Users/joaquingomez/Downloads/Experimentacion/models/word2vec/spacy.word2vec.model"
fastTextModelSpaCy="/Users/joaquingomez/Downloads/Experimentacion/models/fastText/spacy.fastText.model"
GloVeModelSpaCy="/Users/joaquingomez/Downloads/Experimentacion/models/GloVe/spacy.GloVe.model"


# ------------- EXPERIMENTS -------------

#conda activate tfg

# ----- word2vec -----

echo "word2vec running"
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_6_Org -r $DIR_180_6_Dest > $DIR_180_6_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_168_Org -r $DIR_180_168_Dest > $DIR_180_168_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_579_Org -r $DIR_180_579_Dest > $DIR_180_579_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_735_Org -r $DIR_180_735_Dest > $DIR_180_735_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_836_Org -r $DIR_180_836_Dest > $DIR_180_836_Dest/results_word2vec.txt

python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_185_Org -r $DIR_180_185_Dest > $DIR_180_185_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_194_Org -r $DIR_180_194_Dest > $DIR_180_194_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_206_Org -r $DIR_180_206_Dest > $DIR_180_206_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_207_Org -r $DIR_180_207_Dest > $DIR_180_207_Dest/results_word2vec.txt
python3 experiment1-small_text.py -t word2vec -m $word2vecModel -s $word2vecModelSpaCy -d $DIR_180_235_Org -r $DIR_180_235_Dest > $DIR_180_235_Dest/results_word2vec.txt
echo "\n"


# ----- fastText -----

echo "fastText running"
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_6_Org -r $DIR_180_6_Dest > $DIR_180_6_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_168_Org -r $DIR_180_168_Dest > $DIR_180_168_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_579_Org -r $DIR_180_579_Dest > $DIR_180_579_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_735_Org -r $DIR_180_735_Dest > $DIR_180_735_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_836_Org -r $DIR_180_836_Dest > $DIR_180_836_Dest/results_fastText.txt

python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_185_Org -r $DIR_180_185_Dest > $DIR_180_185_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_194_Org -r $DIR_180_194_Dest > $DIR_180_194_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_206_Org -r $DIR_180_206_Dest > $DIR_180_206_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_207_Org -r $DIR_180_207_Dest > $DIR_180_207_Dest/results_fastText.txt
python3 experiment1-small_text.py -t fastText -m $fastTextModel -s $fastTextModelSpaCy -d $DIR_180_235_Org -r $DIR_180_235_Dest > $DIR_180_235_Dest/results_fastText.txt
echo "\n"


# ----- GloVe -----

echo "GloVe running"
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_6_Org -r $DIR_180_6_Dest > $DIR_180_6_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_168_Org -r $DIR_180_168_Dest > $DIR_180_168_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_579_Org -r $DIR_180_579_Dest > $DIR_180_579_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_735_Org -r $DIR_180_735_Dest > $DIR_180_735_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_836_Org -r $DIR_180_836_Dest > $DIR_180_836_Dest/results_GloVe.txt

python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_185_Org -r $DIR_180_185_Dest > $DIR_180_185_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_194_Org -r $DIR_180_194_Dest > $DIR_180_194_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_206_Org -r $DIR_180_206_Dest > $DIR_180_206_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_207_Org -r $DIR_180_207_Dest > $DIR_180_207_Dest/results_GloVe.txt
python3 experiment1-small_text.py -t GloVe -m $GloVeModel -s $GloVeModelSpaCy -d $DIR_180_235_Org -r $DIR_180_235_Dest > $DIR_180_235_Dest/results_GloVe.txt
echo "\n"


# ----- doc2vec -----

echo "doc2vec running"
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NON -d $DIR_180_6_Org -r $DIR_180_6_Dest > $DIR_180_6_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_168_Org -r $DIR_180_168_Dest > $DIR_180_168_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_579_Org -r $DIR_180_579_Dest > $DIR_180_579_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_735_Org -r $DIR_180_735_Dest > $DIR_180_735_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_836_Org -r $DIR_180_836_Dest > $DIR_180_836_Dest/results_doc2vec.txt

python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_185_Org -r $DIR_180_185_Dest > $DIR_180_185_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_194_Org -r $DIR_180_194_Dest > $DIR_180_194_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_206_Org -r $DIR_180_206_Dest > $DIR_180_206_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_207_Org -r $DIR_180_207_Dest > $DIR_180_207_Dest/results_doc2vec.txt
python3 experiment1-small_text.py -t doc2vec -m $doc2vecModel -s NONE -d $DIR_180_235_Org -r $DIR_180_235_Dest > $DIR_180_235_Dest/results_doc2vec.txt
echo "\n"


# ----- ELMO -----

echo "ELMo running"

echo "\n"
