# Process data
bash vqa_tools/download.sh
python vqa_tools/create_dictionary.py
python vqa_tools/compute_softscore.py
# python vqa_tools/detection_features_converter.py
