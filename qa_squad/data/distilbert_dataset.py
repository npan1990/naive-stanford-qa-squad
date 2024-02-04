import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SquadDataset(Dataset):
    def __init__(self, squad_dataset, tokenizer):
        self.tokenizer = tokenizer

        trainset_d = json.load(open(squad_dataset))
        trainset_df = pd.DataFrame(trainset_d['data'])
        trainset_df['paragraphs_number'] = trainset_df['paragraphs'].str.len()
        trainset_df['doc_id'] = range(1, trainset_df.shape[0] + 1)
        paragraphs_trainset_df = trainset_df.explode(['paragraphs'])
        paragraphs_trainset_df['paragraph_id'] = range(1, paragraphs_trainset_df.shape[0] + 1)
        paragraphs_trainset_df['qa'] = paragraphs_trainset_df.apply(lambda x: x['paragraphs']['qas'], axis=1)
        paragraphs_trainset_df['context'] = paragraphs_trainset_df.apply(lambda x: x['paragraphs']['context'], axis=1)
        paragraphs_trainset_df['number_qa'] = paragraphs_trainset_df['qa'].str.len()

        q_trainset_df = paragraphs_trainset_df.explode(['qa'])
        q_trainset_df['question'] = q_trainset_df.apply(lambda x: x['qa']['question'], axis=1)
        q_trainset_df['answers'] = q_trainset_df.apply(lambda x: x['qa']['answers'], axis=1)
        q_trainset_df['answers_number'] = q_trainset_df['answers'].str.len()

        qa_trainset_df = q_trainset_df.explode(['answers'])

        positive_qa_df = qa_trainset_df[~qa_trainset_df['answers'].isnull()].copy()
        positive_qa_df['answer'] = positive_qa_df.apply(lambda x: x['answers']['text'], axis=1)
        positive_qa_df['answer_start'] = positive_qa_df.apply(lambda x: x['answers']['answer_start'], axis=1)
        positive_qa_df['answer_end'] = positive_qa_df.apply(
            lambda x: x['answers']['answer_start'] + len(x['answers']['text']), axis=1)
        self.positive_qa_df = positive_qa_df
        self.rows = []

        self.preprocess()

    def preprocess(self):
        for index, row in tqdm(self.positive_qa_df.iterrows(), total=self.positive_qa_df.shape[0]):
            context = row['context']
            correct_answer = row['answer']
            question = row['question']
            answer_start = row['answer_start']
            answer_end = row['answer_end']

            start = -1
            end = -1
            context_question_encoding = self.tokenizer(question, context, return_offsets_mapping=True,
                                                       padding='max_length',
                                                       truncation=True)
            offset_mapping = context_question_encoding.pop("offset_mapping")
            for i, span in enumerate(offset_mapping):
                if span[0] <= answer_start <= span[1]:
                    start = i
                if span[0] <= answer_end <= span[1]:
                    end = i

                if start != -1 and end != -1:
                    break
            if start != -1 and end != -1:
                row = {
                    'input_ids': torch.tensor(context_question_encoding['input_ids']),
                    'attention_mask': torch.tensor(context_question_encoding['attention_mask']),
                    'start_positions': torch.tensor(start),
                    'end_positions': torch.tensor(end)}
                self.rows.append(row)

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)
