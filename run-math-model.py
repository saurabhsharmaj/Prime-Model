import joblib

# Important: make PrimeModel available
from prime_model import PrimeModel


# -------------------------
# LOAD MODEL
# -------------------------

model = joblib.load("prime_math_model.pkl")

print("Model loaded!")


# -------------------------
# PREDICT
# -------------------------

number = int(input("Enter a number: "))

result = model.predict(number)

print(number, "=>", result)