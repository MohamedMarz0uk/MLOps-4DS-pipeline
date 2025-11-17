# Variables
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQUIREMENTS := requirements.txt

# ----------------------------
# 1. Installation
# ----------------------------
install: $(VENV)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	@echo "Environnement virtuel créé"

# ----------------------------
# 2. Préparer les données
# ----------------------------
prepare:
	$(PYTHON) main.py --prepare

# ----------------------------
# 3. Entraîner le modèle
# ----------------------------
train:
	$(PYTHON) main.py --train

# ----------------------------
# 4. Évaluer le modèle
# ----------------------------
evaluate:
	$(PYTHON) main.py --evaluate

# ----------------------------
# 5. Générer des prédictions / fichier de soumission
# ----------------------------
predict:
	$(PYTHON) main.py --predict

# ----------------------------
# 6. Lancer les tests rapides
# ----------------------------
test:
	$(PYTHON) test_pipeline.py

# ----------------------------
# 7. Vérification du code (CI)
# ----------------------------
lint:
	flake8 *.py
	black --check *.py
	bandit -r .

format:
	black *.py
	isort *.py

# ----------------------------
# 8. Nettoyage des fichiers générés
# ----------------------------
clean:
	rm -rf $(VENV)
	rm -f *.pkl
	rm -f *.csv
	rm -rf _pycache_

# ----------------------------
# 9. Tout exécuter (pipeline complet)
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