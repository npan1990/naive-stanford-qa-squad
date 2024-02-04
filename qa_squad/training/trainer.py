import os
import pickle

import torch
from loguru import logger
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import click
import numpy as np

from qa_squad.config.configuration import GLOVE_EMBEDDINGS_100_WORDS_TXT
from qa_squad.data.bidaf_datasets import SquadDataset
from qa_squad.models.bidaf import BiDAF
from qa_squad.models.tokenizers import BiDAFTokenizer

"""
A custom trainer for demo purposes.
Normally this approach must be done with a commercial solution such as huggingface.
"""


class BiDAFTrainer:

    def __init__(self, processed_dataset: str, eval_dataset:str, word_tokenizer: BiDAFTokenizer, char_tokenizer: BiDAFTokenizer,
                 epochs: int = 1, device: str = 'cuda:0', batch_size: int = 4,
                 glove_embeddings: str = GLOVE_EMBEDDINGS_100_WORDS_TXT,
                 model_folder_save: str = './', sample: int = 0):
        """
        BiDAF Trainer
        Args:
            processed_dataset(str): The filename of the processed dataset.
            eval_dataset(str): The filename of the eval dataset.
            word_tokenizer(str): The filename of the word tokenizer.
            char_tokenizer(str): The filename of the char tokenizer.
            epochs(int): The number of epochs.
        """
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer

        CHAR_VOCAB_DIM = len(self.char_tokenizer.word2idx)
        EMB_DIM = 100
        CHAR_EMB_DIM = 8
        NUM_OUTPUT_CHANNELS = 50
        KERNEL_SIZE = (8, 5)
        HIDDEN_DIM = 100
        self.device = torch.device(device)
        self.epochs = epochs
        self.model_folder_save = model_folder_save

        self.model = BiDAF(CHAR_VOCAB_DIM,
                           EMB_DIM,
                           CHAR_EMB_DIM,
                           NUM_OUTPUT_CHANNELS,
                           KERNEL_SIZE,
                           HIDDEN_DIM,
                           device, word_tokenizer=word_tokenizer,
                           glove_embeddings=glove_embeddings).to(device)
        self.optimizer = optim.Adadelta(self.model.parameters())

        self.dataset = SquadDataset(processed_dataset, batch_size=batch_size, sample=sample)
        self.evaluation_dataset = SquadDataset(eval_dataset, batch_size=batch_size, sample=sample)

    def train(self):
        for epoch in range(self.epochs):
            logger.info(f'Running Epoch - {epoch+1}.')
            train_loss = 0
            steps = 0
            self.model = self.model.train()
            for batch in tqdm(self.dataset, total=len(self.dataset)):
                try:
                    steps+=1
                    self.optimizer.zero_grad()

                    word_padded_context, question_padded_context, char_padded_context, char_padded_question, label, tokenized_answer, tokenized_context = batch
                    word_padded_context = word_padded_context.to(self.device)
                    question_padded_context = question_padded_context.to(self.device)
                    char_padded_context = char_padded_context.to(self.device)
                    char_padded_question = char_padded_question.to(self.device)
                    label = label.to(self.device)

                    preds = self.model(word_padded_context, question_padded_context, char_padded_context,
                                       char_padded_question)

                    start_pred, end_pred = preds
                    s_idx, e_idx = label[:, 0], label[:, 1]

                    loss = (F.cross_entropy(start_pred, s_idx) + F.cross_entropy(end_pred, e_idx)) / 2.0

                    loss.backward()

                    self.optimizer.step()

                    train_loss += loss.item()

                    if steps % 100 == 0:
                        current_loss = train_loss / steps
                        logger.warning(f'Current Loss - {current_loss}')
                except:
                    logger.critical(f'Skipped step: {steps}.')
            torch.save(self.model.state_dict(), os.path.join(self.model_folder_save, f'bidaf_{epoch}.bin'))
            loss = train_loss / len(self.dataset)
            logger.success(f'Loss - {loss}.')
            val_loss, f1, em = self.eval()
            logger.success(f'Val Loss: {val_loss}, F1: {f1}, EM: {em}')

    def eval(self):
        valid_loss = 0.
        batch_count = 0

        self.model = self.model.eval()
        predictions = {}
        predicted_answers = []
        correct_answers = []
        for batch in self.evaluation_dataset:

            batch_count += 1

            word_padded_context, question_padded_context, char_padded_context, char_padded_question, label, tokenized_answer, tokenized_context = batch
            word_padded_context = word_padded_context.to(self.device)
            question_padded_context = question_padded_context.to(self.device)
            char_padded_context = char_padded_context.to(self.device)
            char_padded_question = char_padded_question.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                preds = self.model(word_padded_context, question_padded_context, char_padded_context,
                                   char_padded_question)

                s_idx, e_idx = label[:, 0], label[:, 1]

                p1, p2 = preds

                try:
                    loss = (F.cross_entropy(p1, s_idx) + F.cross_entropy(p2, e_idx)) / 2.0
                except:
                    # TODO: Fix the case where the last bast contains only 1 item
                    logger.critical(f'Unable to eval batch {batch_count}. Batch will not be used for eval.')
                    continue

                valid_loss += loss.item()

                # convert predictions to text

                start_indices = p1.argmax(axis=1).cpu().numpy()
                end_indices = p2.argmax(axis=1).cpu().numpy()
                indices = np.column_stack([start_indices, end_indices])

                for i in range(len(batch)):
                    try:
                        correct_answer = tokenized_answer.iloc[i]
                        predicted_answer = np.array(tokenized_context)[i][indices[i, 0]:indices[i, 1]]

                        predicted_answers.append(predicted_answer)
                        correct_answers.append(correct_answer)
                    except:
                        # The predicted answer is not within context
                        continue

        f1, em = self.compute_all_metrics(predicted_answers, correct_answers)
        return valid_loss / len(self.evaluation_dataset), em, f1

    def compute_all_metrics(self, predicted_answers, correct_answers):
        f1s = []
        ems = []
        for i in range(len(predicted_answers)):
            predicted_answer = predicted_answers[i]
            correct_answer = correct_answers[i]

            predicted_answer_s = set(predicted_answer)
            correct_answer_s = set(correct_answer)

            tp = float(len(predicted_answer_s & correct_answer_s))
            fp = float(len(predicted_answer_s - correct_answer_s))
            fn = float(len(correct_answer_s - predicted_answer_s))

            try:
                pr = tp / (tp+fp)
                rc = tp / len(correct_answer_s)
                f1 = 2*pr*rc/(pr+rc)
            except ZeroDivisionError:
                f1 = 0.0

            f1s.append(f1)

            if predicted_answer_s == correct_answer_s:
                ems.append(1)
            else:
                ems.append(0)

        return np.average(f1s), np.average(ems)


@click.command()
@click.option('--processed_dataset', default='./data/squad/trainset.pickle', help='The processed dataset.')
@click.option('--eval_dataset', default='./data/squad/testset.pickle', help='The processed dataset.')
@click.option('--word_tokenizer', default='./data/squad/word_tokenizer.pickle', help='The word tokenizer.')
@click.option('--char_tokenizer', default='./data/squad/char_tokenizer.pickle', help='The char tokenizer.')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--batch_size', default=4, help='The batch size.')
@click.option('--model_folder_save', default='./data/squad/', help='The batch size.')
@click.option('--sample', type=click.INT, default=10000, help='Sample from trainset.')
@click.option('--device', type=click.STRING, default='cuda:0', help='The device.')
def qa_train(processed_dataset: str, eval_dataset, word_tokenizer: str, char_tokenizer: str, epochs: int,
             batch_size: int, model_folder_save: str, sample: int, device: str):
    word_tokenizer = pickle.load(open(word_tokenizer, 'rb'))
    char_tokenizer = pickle.load(open(char_tokenizer, 'rb'))
    simple_trainer = BiDAFTrainer(processed_dataset=processed_dataset,
                                  eval_dataset=eval_dataset,
                                  word_tokenizer=word_tokenizer,
                                  char_tokenizer=char_tokenizer,
                                  epochs=epochs, batch_size=batch_size,
                                  model_folder_save=model_folder_save,
                                  sample=sample,
                                  device=device)
    simple_trainer.train()


# TODO: Replace with test
if __name__ == '__main__':
    qa_train()
