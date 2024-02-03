import pickle
import torch
import numpy as np

from loguru import logger
from tqdm import tqdm

from qa_squad.config.configuration import MIN_LENGTH


class SquadDataset:
    """
    Custom dataset class that returns the tensors required by the BDAF model.
    Each row must contain:

        1. tokenized context, answer, question
        2. The token ids of the correct answer
    """

    def __init__(self, dataset_pickle: str,
                 batch_size: int = 4, sample: int = 0):
        """
        Prepared a Squad dataset for the BiDAF network.
        The batches much contain:
            * word context, question
            * char context, question
            * labels

        All the batches are padded using the PAD token.

        Args:
            dataset_pickle(pd.DataFrame): The preprocessed dataset.
            batch_size(int): The batch size.
            sample(int): Use only sample examples.
        """
        self.batch_size = batch_size

        logger.info('Reading datasets.')
        dataset = pickle.load(open(dataset_pickle, 'rb'))

        if sample > 0:
            dataset = dataset.sample(sample)

        dataset = dataset[(dataset['label_start_token_index']>=0) & (dataset['label_end_token_index']>=0)]
        batches = [dataset[i:i + self.batch_size] for i in range(0, len(dataset), self.batch_size)]
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:

            # We need to pad to max sizes
            max_context_words_length = batch['context_word_ids'].str.len().max()
            if max_context_words_length < MIN_LENGTH:
                max_context_words_length = MIN_LENGTH

            max_answer_words_length = batch['answer_word_ids'].str.len().max()
            max_question_words_length = batch['question_word_ids'].str.len().max()
            if max_question_words_length < MIN_LENGTH:
                max_question_words_length = MIN_LENGTH

            max_context_chars_length = batch['context_char_ids'].map(np.concatenate).str.len().max()
            if max_context_chars_length < MIN_LENGTH:
                max_context_chars_length = MIN_LENGTH
            max_answer_chars_length = batch['answer_char_ids'].map(np.concatenate).str.len().max()
            max_question_chars_length = batch['question_char_ids'].map(np.concatenate).str.len().max()
            if max_question_chars_length < MIN_LENGTH:
                max_question_chars_length = MIN_LENGTH

            # word padded context
            # Token id 1 is PAD
            # Dimension is BatchSize x Max Context Tokens
            word_padded_context = torch.LongTensor(len(batch), max_context_words_length).fill_(1)
            for i, ctx in enumerate(batch.context_word_ids):
                word_padded_context[i, :len(ctx)] = torch.LongTensor(ctx)

            # word padded question
            # Token id 1 is PAD
            # Dimension is BatchSize x Max Question Tokens
            question_padded_context = torch.LongTensor(len(batch), max_question_words_length).fill_(1)
            for i, ctx in enumerate(batch.question_word_ids):
                question_padded_context[i, :len(ctx)] = torch.LongTensor(ctx)

            # Char padded context
            # Dimension is BatchSize x Max Word Tokens x Max Char Tokens
            char_padded_context = torch.ones(len(batch), max_context_words_length, max_context_chars_length).type(torch.LongTensor)
            for batch_i, context in enumerate(batch.context_char_ids):
                # Iterate words. Each word has a char vector.
                for word_i, char_vector in enumerate(context):
                    for char_i, char in enumerate(char_vector):
                        char_padded_context[batch_i, word_i, char_i] = char

            # Char padded question
            # Dimension is BatchSize x Max Question Tokens x Max Char Tokens
            char_padded_question = torch.ones(len(batch), max_question_words_length,
                                             max_question_chars_length).type(torch.LongTensor)
            for batch_i, context in enumerate(batch.question_char_ids):
                # Iterate words. Each word has a char vector.
                for word_i, char_vector in enumerate(context):
                    for char_i, char in enumerate(char_vector):
                        char_padded_question[batch_i, word_i, char_i] = char

            label = torch.LongTensor(batch[['label_start_token_index', 'label_end_token_index']].values)

            # only for the evaluation
            tokenized_answer = batch.tokenized_answer
            tokenized_context = batch.tokenized_context

            yield (word_padded_context, question_padded_context, char_padded_context, char_padded_question, label,
                   tokenized_answer, tokenized_context)


# TODO: Replace with test
if __name__ == '__main__':
    squad = SquadDataset('./data/squad/trainset.pickle')
    for batch in tqdm(squad, total=len(squad)):
        continue
