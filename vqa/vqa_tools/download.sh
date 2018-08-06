## Script for downloading data

# GloVe Vectors
wget -P vqa_data http://nlp.stanford.edu/data/glove.6B.zip
unzip vqa_data/glove.6B.zip -d vqa_data/glove
# rm vqa_data/glove.6B.zip

# Questions
wget -P vqa_data http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip vqa_data/v2_Questions_Train_mscoco.zip -d vqa_data
rm vqa_data/v2_Questions_Train_mscoco.zip

wget -P vqa_data http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip vqa_data/v2_Questions_Val_mscoco.zip -d vqa_data
rm vqa_data/v2_Questions_Val_mscoco.zip

wget -P vqa_data http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip vqa_data/v2_Questions_Test_mscoco.zip -d vqa_data
rm vqa_data/v2_Questions_Test_mscoco.zip

# Annotations
wget -P vqa_data http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip vqa_data/v2_Annotations_Train_mscoco.zip -d vqa_data
rm vqa_data/v2_Annotations_Train_mscoco.zip

wget -P vqa_data http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip vqa_data/v2_Annotations_Val_mscoco.zip -d vqa_data
rm vqa_data/v2_Annotations_Val_mscoco.zip

# Image Features
wget -P vqa_data https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip vqa_data/trainval_36.zip -d vqa_data
rm vqa_data/trainval_36.zip
