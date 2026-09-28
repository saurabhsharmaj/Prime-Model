import joblib

from prime_model import PrimeModel


# -------------------------
# CREATE MODEL
# -------------------------

model = PrimeModel()

print("Mathematical model created!")


# -------------------------
# TEST DATA
# -------------------------

test_data = [
    (17, "PRIME"),
    (19, "PRIME"),
    (21, "NOT PRIME"),
    (22, "NOT PRIME"),
    (23, "PRIME"),
    (24, "NOT PRIME"),
    (25, "NOT PRIME"),
    (29, "PRIME"),
    (31, "PRIME"),
    (35, "NOT PRIME")
]


# -------------------------
# TEST MODEL
# -------------------------

correct = 0

for number, actual in test_data:

    predicted = model.predict(number)

    print(
        number,
        "Actual =", actual,
        "Predicted =", predicted
    )

    if actual == predicted:
        correct += 1


accuracy = correct / len(test_data)

print()
print("Accuracy =", accuracy * 100, "%")


# -------------------------
# EXPORT MODEL
# -------------------------

joblib.dump(model, "prime_math_model.pkl")

print("Model exported!")