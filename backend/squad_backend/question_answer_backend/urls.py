from django.urls import path

from . import views

app_name = "question_answer_backend"
urlpatterns = [
    path("", views.index, name="index"),
    path("about", views.index, name="index"),
    path("try", views.qa_try, name="try"),
]
