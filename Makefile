train:
	python src/train.py data/imdb.csv model.joblib

eval:
	python src/eval.py

serve:
	uvicorn app.main:app --reload

dvc-run:
	dvc repro