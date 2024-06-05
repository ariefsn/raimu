run.dev:
	fastapi dev app/main.py

freeze:
	pip3 freeze > requirements.txt

install:
	pip3 install -r ./requirements.txt

clean:
	rm -rf ./dev

env.create:
	python3.12 -m venv dev

env.activate:
	source ./dev/bin/activate