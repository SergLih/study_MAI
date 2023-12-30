#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting Postgres..."

    while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
      sleep 0.1
    done

    echo "Postgres launched"
fi

exec "$@"
