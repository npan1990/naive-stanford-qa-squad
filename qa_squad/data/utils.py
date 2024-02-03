import os
import json
import re
from typing import List, Tuple

import pickle
import pandas as pd
import spacy
import click
from loguru import logger
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from qa_squad.config.configuration import PROCESSED_SQUAD_DATASET_DIR
from qa_squad.models.tokenizers import BiDAFTokenizer

MAXIMUM_CONTEXT = 1024


def build_word_vectorizer(documents: List[List[str]]) -> BiDAFTokenizer:
    """
    Map each word to an id.

    Args:
        documents(List): List of documents. Each document is a list of terms.

    Returns:
        Tuple
    """

    words = set()
    for document in tqdm(documents):
        for word in document:
                words.add(word)

    words = ['<unk>', '<pad>'] + list(words)

    word_vocab = set(words)
    word_d = {i: word for i, word in enumerate(words)}
    word2idx = {word: idx for idx, word in word_d.items()}
    idx2word = {v: k for k, v in word2idx.items()}

    return BiDAFTokenizer(word_vocab, word2idx, idx2word)


def build_char_vectorizer(documents: List) -> BiDAFTokenizer:
    """
    Map each character to an id.

    Args:
        documents(List): List of documents. Each document is a list of terms.

    Returns:
        Tuple
    """

    chars = set()
    for document in tqdm(documents):
        for word in document:
            for char in word:
                chars.add(char)

    char_vocab = set(chars)
    char_d = {i: char for i, char in enumerate(chars)}
    char2idx = {char: idx for idx, char in char_d.items()}
    idx2char = {v: k for k, v in char2idx.items()}

    return BiDAFTokenizer(char_vocab, char2idx, idx2char,char_tokenizer=True)


def normalize_text(text: str) -> str:
    """
    Normalize content.

    Args:
        text(str): Content

    Returns:

    """
    text = text.lower()

    # replace every whitespace with single space
    text = re.sub(r'[\s+]', ' ', text)

    # remove weird characters but keep stops and commas
    text = re.sub(r'[^\w ,\.\(\):;\%?]', ' ', text)

    return text


class DataProcessor:

    def __init__(self, directory: str):
        """

        Args:
            directory(str): Dataset directory. It must contain, train-v2.0.json and dev-v2.0.json.
        """
        train_file = os.path.join(directory, 'train-v2.0.json')
        dev_file = os.path.join(directory, 'dev-v2.0.json')
        trainset_d = json.load(open(train_file))
        devset_d = json.load(open(dev_file))

        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = Tokenizer(self.nlp.vocab)

        self.trainset_df = pd.DataFrame(trainset_d['data'])
        self.devset_df = pd.DataFrame(devset_d['data'])

        self.trainset_qa = None
        self.devset_qa = None
        self.word_tokenizer = None
        self.char_tokenizer = None

    def prepare(self) -> None:
        """
        Prepare datasets:

            * Create QA Pairs
            * Normalize text
            *Tokenize text

        Returns:
            None
        """
        logger.info('Prepare Trainset.')
        self.trainset_qa, corpus_train = self.prepare_qa_dataset(self.trainset_df)
        logger.info('Prepare Devset.')
        self.devset_qa, corpus_dev = self.prepare_qa_dataset(self.devset_df)

        total_corpus = corpus_train + corpus_dev

        logger.info('Creating tokenizers.')
        self.word_tokenizer = build_word_vectorizer(total_corpus)
        self.char_tokenizer = build_char_vectorizer(total_corpus)
        logger.success('Created BiDAF-Tokenizers.')

        self.trainset_qa['answer_word_ids'] = self.trainset_qa['tokenized_answer'].map(self.word_tokenizer.transform)
        self.trainset_qa['context_word_ids'] = self.trainset_qa['tokenized_context'].map(self.word_tokenizer.transform)
        self.trainset_qa['question_word_ids'] = self.trainset_qa['tokenized_question'].map(self.word_tokenizer.transform)

        self.trainset_qa['answer_char_ids'] = self.trainset_qa['tokenized_answer'].map(self.char_tokenizer.transform)
        self.trainset_qa['context_char_ids'] = self.trainset_qa['tokenized_context'].map(self.char_tokenizer.transform)
        self.trainset_qa['question_char_ids'] = self.trainset_qa['tokenized_question'].map(
            self.char_tokenizer.transform)

        self.devset_qa['answer_word_ids'] = self.devset_qa['tokenized_answer'].map(self.word_tokenizer.transform)
        self.devset_qa['context_word_ids'] = self.devset_qa['tokenized_context'].map(self.word_tokenizer.transform)
        self.devset_qa['question_word_ids'] = self.devset_qa['tokenized_question'].map(
            self.word_tokenizer.transform)

        self.devset_qa['answer_char_ids'] = self.devset_qa['tokenized_answer'].map(self.char_tokenizer.transform)
        self.devset_qa['context_char_ids'] = self.devset_qa['tokenized_context'].map(self.char_tokenizer.transform)
        self.devset_qa['question_char_ids'] = self.devset_qa['tokenized_question'].map(
            self.char_tokenizer.transform)

        self.store_datasets()

    def store_datasets(self) -> None:
        """
        Store dataset to default dir

        Returns:
            None
        """
        os.makedirs(PROCESSED_SQUAD_DATASET_DIR, exist_ok=True)
        logger.info('Storing vectorizers.')
        pickle.dump(self.word_tokenizer,
                    open(os.path.join(PROCESSED_SQUAD_DATASET_DIR, 'word_tokenizer.pickle'), 'wb'))
        pickle.dump(self.char_tokenizer,
                    open(os.path.join(PROCESSED_SQUAD_DATASET_DIR, 'char_tokenizer.pickle'), 'wb'))

        logger.info('Storing processed dataset')
        pickle.dump(self.trainset_qa,
                    open(os.path.join(PROCESSED_SQUAD_DATASET_DIR, 'trainset.pickle'), 'wb'))
        pickle.dump(self.devset_qa,
                    open(os.path.join(PROCESSED_SQUAD_DATASET_DIR, 'testset.pickle'), 'wb'))
        logger.success('Dataset is prepared. Use these files for Machine Learning.')

    def prepare_qa_dataset(self, df: pd.DataFrame) -> Tuple:
        """
        Create the appropriate features for a squad row.
        Args:
            df: Original squad data frame.

        Returns:
            Tuple
        """
        df['paragraphs_number'] = df['paragraphs'].str.len()
        df['doc_id'] = range(1, df.shape[0] + 1)

        corpus = []
        logger.debug('Processing context')
        paragraphs_df = df.explode(['paragraphs'])
        paragraphs_df['paragraph_id'] = range(1, paragraphs_df.shape[0] + 1)
        paragraphs_df['qa'] = paragraphs_df.apply(lambda x: x['paragraphs']['qas'], axis=1)
        paragraphs_df['context'] = paragraphs_df.apply(lambda x: x['paragraphs']['context'], axis=1)
        paragraphs_df = paragraphs_df[paragraphs_df['context'].str.len()<=MAXIMUM_CONTEXT]
        paragraphs_df['number_qa'] = paragraphs_df['qa'].str.len()
        paragraphs_df['context'] = paragraphs_df['context'].map(normalize_text)
        paragraphs_df['tokenized_context'] = paragraphs_df['context'].map(self.tokenize_content)
        paragraphs_df['tokenized_context_spans'] = paragraphs_df['context'].map(self.tokenize_content_spans)
        corpus.extend(paragraphs_df['tokenized_context'].tolist())

        logger.debug('Processing questions')
        q_df = paragraphs_df.explode(['qa'])
        q_df['question'] = q_df.apply(lambda x: x['qa']['question'], axis=1)
        q_df['answers'] = q_df.apply(lambda x: x['qa']['answers'], axis=1)
        q_df['answers_number'] = q_df['answers'].str.len()
        q_df['question'] = q_df['question'].map(normalize_text)
        q_df['tokenized_question'] = q_df['question'].map(self.tokenize_content)
        corpus.extend(q_df['tokenized_question'].tolist())

        logger.debug('Processing answers')
        qa_df = q_df.explode(['answers'])
        positive_qa_df = qa_df[~qa_df['answers'].isnull()].copy()
        positive_qa_df['answer'] = positive_qa_df.apply(lambda x: x['answers']['text'], axis=1)
        positive_qa_df['answer_start'] = positive_qa_df.apply(lambda x: x['answers']['answer_start'], axis=1)
        positive_qa_df['answer_end'] = positive_qa_df.apply(lambda x: x['answers']['answer_start']+len(x['answers']['text']), axis=1)
        positive_qa_df['answer'] = positive_qa_df['answer'].map(normalize_text)
        positive_qa_df['tokenized_answer'] = positive_qa_df['answer'].map(self.tokenize_content)
        corpus.extend(positive_qa_df['tokenized_answer'].tolist())

        labels_start_token_index = []
        labels_end_token_index = []
        # Identify the answer token indices
        for index, row in tqdm(positive_qa_df.iterrows(), total=positive_qa_df.shape[0]):
            answer_start = row['answer_start']
            answer_end = row['answer_end']
            context_spans = row['tokenized_context_spans']

            start = -1
            end = -1
            for i, span in enumerate(context_spans):
                if answer_start >=span[0] and answer_start<span[1]:
                    start = i

                if answer_end <= span[1] and answer_end > span[0]:
                    end = i
            labels_start_token_index.append(start)
            labels_end_token_index.append(end)

        positive_qa_df['label_start_token_index'] = labels_start_token_index
        positive_qa_df['label_end_token_index'] = labels_end_token_index

        return positive_qa_df, corpus

    def tokenize_content(self, content) -> List[str]:
        """
        Tokenize string using spacy tokenizers.

        Args:
            content(str): The content.

        Returns:
            List[str]
        """
        words = []
        for word in self.tokenizer(content):
            # Store token, start, end position
            words.append(word.text)

        return words

    def tokenize_content_spans(self, content) -> List[Tuple]:
        """
        Tokenize and also just return spans.

        Args:
            content(str): The content.

        Returns:
            List[str]
        """
        spans = []
        for word in self.tokenizer(content):
            # Store token, start, end position
            spans.append((word.idx, word.idx+len(word.text)))

        return spans


@click.command()
@click.option('--directory', default='./data/squad/', help='directory')
def create_dataset(directory):
    """
    Create a dataset for machine learning.
    Args:
        directory:

    Returns:

    """
    processor = DataProcessor(directory)
    processor.prepare()

# TODO: Replace with test
if __name__ == '__main__':
    create_dataset()
