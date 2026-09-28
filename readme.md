python -m venv .venv
source .venv/Scripts/activate

python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org



python train-save-math-model.py
python run-math-model.py