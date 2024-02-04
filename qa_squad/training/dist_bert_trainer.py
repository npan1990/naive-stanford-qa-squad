import os

from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import click
from loguru import logger
import torch

from qa_squad.data.distilbert_dataset import SquadDataset

HUB_TOKEN = os.getenv('HUB_TOKEN', '')


class DistilBERTTrainer:

    def __init__(self, trainset: str, devset: str, max_length: int = 1024, checkpoint: str = "distilbert-base-uncased",
                 lr: float = 1e-4, batch_size: int = 8, organization: str = "",
                 saved_model: str = 'qa', epochs: int = 5):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding="max_length", max_length=max_length,
                                                       truncation=True)

        logger.info('Loading train and development sets.')
        self.squad_trainset = SquadDataset(trainset, self.tokenizer)
        self.squad_devset = SquadDataset(devset, self.tokenizer)

        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, torch_dtype=torch.float32)

        self.training_args = TrainingArguments(

            output_dir="test",
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            push_to_hub=True,
            hub_token=HUB_TOKEN,
            push_to_hub_organization=organization,
            push_to_hub_model_id=saved_model,
            report_to=["tensorboard"],
            save_steps=500
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.squad_trainset,
            eval_dataset=self.squad_devset,
            data_collator=DefaultDataCollator(return_tensors='pt')
        )

    def train(self):
        self.trainer.train()
        self.trainer.push_to_hub()


@click.command()
@click.option('--processed_dataset', default='./data/squad/train-v2.0.json', help='The processed dataset.')
@click.option('--eval_dataset', default='./data/squad/dev-v2.0.json', help='The processed dataset.')
@click.option('--epochs', default=2, help='Epochs')
@click.option('--batch_size', default=8, help='The batch size.')
@click.option('--organization', default='npan1990', help='The company.')
@click.option('--saved_model', type=click.STRING, default='squad-distil-bert', help='Model alias.')
def distil_bert_qa_train(processed_dataset: str, eval_dataset: str,
                         epochs: int, batch_size: int, organization: str, saved_model: str):
    distil_bert_trainer = DistilBERTTrainer(processed_dataset, eval_dataset, epochs=epochs, batch_size=batch_size,
                                            organization=organization, saved_model=saved_model)
    distil_bert_trainer.train()


if __name__ == '__main__':
    distil_bert_qa_train()
