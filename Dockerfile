FROM python:3.8.8

ENV PYTHONUNBUFFERED 1

RUN mkdir /app
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
COPY . /app
RUN chmod +x /app/start.sh
CMD ["python", "main.py"]