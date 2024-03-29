import os

PROCESSED_SQUAD_DATASET_DIR = os.getenv('PROCESSED_SQUAD_DATASET_DIR', 'data/squad/')
GLOVE_EMBEDDINGS_100_WORDS_TXT = os.getenv('GLOVE_EMBEDDINGS_100_WORDS_TXT', 'data/embeddings/glove.6B.100d.txt')
MIN_LENGTH = 100