build:
	docker-compose build
test:
	docker exec -it works_app_1 python3 manage.py test
migrate:
	docker exec -it works_app_1 python3 manage.py flush --no-input
	docker exec -it works_app_1 python3 manage.py makemigrations tennis_application
	docker exec -it works_app_1 python3 manage.py migrate
up:	build
	docker-compose up
down:
	docker-compose down --volumes
static:
	docker exec -it works_app_1 python3 manage.py collectstatic
shell:
	docker exec -it works_app_1 python3 manage.py shell
