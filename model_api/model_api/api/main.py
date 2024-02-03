from typing import List

from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering
import torch

from model_api.config.config import MAX_LENGTH, DEVICE
from model_api.model.squad import SquadQuery, SquadAnswer

app = FastAPI()

# TODO: Hide this in packages.
# Initialize custom QA model
tokenizer_squad_qa = AutoTokenizer.from_pretrained("distilbert-base-uncased", padding="max_length",
                                                   max_length=MAX_LENGTH, truncation=True)
model_squad_qa = AutoModelForQuestionAnswering.from_pretrained("npan1990/test", device_map={'': DEVICE})

pretrained_tokenizer_squad_qa = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased-distilled-squad", padding="max_length",
                                                   max_length=MAX_LENGTH, truncation=True)
pretrained_model_squad_qa = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-cased-distilled"
                                                                          "-squad", device_map={'': DEVICE})


@app.post('/squad_qa/')
async def get_article(query: SquadQuery) -> SquadAnswer:
    context = query.context
    question = query.question
    model = query.model

    if model.lower() == 'trained':
        tokenizer = tokenizer_squad_qa
        model_torch = model_squad_qa
    else:
        tokenizer = pretrained_tokenizer_squad_qa
        model_torch = pretrained_model_squad_qa

    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model_torch(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)

        return SquadAnswer(context=context, question=question, answer=answer)
