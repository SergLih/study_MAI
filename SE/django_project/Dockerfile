FROM python:3.6-alpine

WORKDIR /tennis_app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies
RUN apk update \
    && apk add postgresql-dev gcc python3-dev musl-dev curl libc-dev libffi-dev

RUN pip install --upgrade pip
COPY ./tennis_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./tennis_app .
EXPOSE 8080

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
