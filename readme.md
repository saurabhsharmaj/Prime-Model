python -m venv .venv
source .venv/Scripts/activate

python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org


#PK1 - Label Training
python train-save-model.py
python run-model.py

#PK1 - Mathematical model 100% accurate.
python train-save-math-model.py
python run-math-model.py

#ONNX universal model.
python create_prime_onnx.py