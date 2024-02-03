import json
import os

from django.http import HttpResponse
from django.template import loader
import requests

API_URL = os.getenv('API_URL', 'http://localhost:5000/')


# Create your views here.
def index(request):
    template = loader.get_template("question_answer_backend/index.html")

    context = {
        'page': 'about'
    }

    return HttpResponse(template.render(context, request))


def qa_try(request):
    template = loader.get_template("question_answer_backend/try.html")

    context = request.POST.get("context", "")
    question = request.POST.get("question", "")
    model_id = request.POST.get("model_id", "")

    answer = 'Not yet!'
    if context != "" and question != "" and model_id != "":
        r = requests.post(API_URL + 'squad_qa/', json={'context': context, 'question': question, 'model': model_id})
        try:
            answer = json.loads(r.text)['answer']
        except:
            pass

    context = {
        'page': 'try',
        'answer': answer
    }

    return HttpResponse(template.render(context, request))

