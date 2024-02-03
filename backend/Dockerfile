FROM python:3.8

WORKDIR /backend
COPY . .
RUN apt-get update && \
    pip install django==3.2.23 && \
    pip install requests

EXPOSE 80
CMD python manage.py runserver 0.0.0.0:80