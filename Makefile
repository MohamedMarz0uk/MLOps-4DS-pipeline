VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQUIREMENTS := requirements.txt


install: $(VENV)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "Environnement virtuel créé"


prepare:
	$(PYTHON) main.py --prepare


train:
	$(PYTHON) main.py --train


evaluate:
	$(PYTHON) main.py --evaluate


predict:
	$(PYTHON) main.py --predict


test:
	$(PYTHON) test_pipeline.py

# ----------------------------
# Vérification du code
# ----------------------------
lint:
	flake8 *.py
	black --check *.py

security:
	bandit -r . -ll

format:
	black *.py
	isort *.py

# ----------------------------
# Nettoyage des fichiers générés
# ----------------------------
clean:
	rm -f *.pkl
	rm -f *.csv
	rm -rf _pycache_

# ----------------------------
# Tout exécuter (pipeline complet)
# ----------------------------
all: install prepare train evaluate predict test

# ----------------------------
# 10. Surveiller les fichiers et relancer automatiquement
# ----------------------------
watch:
	@echo "=== Surveiller les fichiers avec watchmedo ==="
	watchmedo shell-command \
        --patterns=".py;.csv" \
        --recursive \
        --command='make all'


.PHONY: install prepare train evaluate predict test lint format clean all

deploy:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

flask:
	FLASK_APP=flask_app.py FLASK_ENV=development $(PYTHON) -m flask run --host=0.0.0.0 --port=5000

mlflow-ui:
	@echo "Démarrage de MLflow UI sur 0.0.0.0:4000 avec SQLite..."
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 4000 &

mlflow-stop:
	@echo "Arrêt de MLflow..."
	@pkill -f "mlflow ui" || echo "MLflow n'était pas en cours d'exécution"
	@echo "✓ MLflow arrêté"