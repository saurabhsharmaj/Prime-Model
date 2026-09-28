import numpy as np
import onnxruntime as ort


# ============================================================
# LOAD ONNX MODEL
# ============================================================

session = ort.InferenceSession(
    "prime_math_model.onnx"
)

print("ONNX model loaded!")


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_prime(number):

    result = session.run(
        ["is_prime"],
        {
            "number": np.array([number], dtype=np.int64)
        }
    )

    prediction = result[0][0]

    if prediction == 1:
        return "PRIME"

    return "NOT PRIME"


# ============================================================
# TEST DATA
# ============================================================

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


# ============================================================
# TEST
# ============================================================

correct = 0

for number, actual in test_data:

    predicted = predict_prime(number)

    print(
        number,
        "Actual =", actual,
        "Predicted =", predicted
    )

    if actual == predicted:
        correct += 1


# ============================================================
# ACCURACY
# ============================================================

accuracy = correct / len(test_data)

print()
print("Accuracy =", accuracy * 100, "%")


# ============================================================
# INDIVIDUAL TEST
# ============================================================

print()
print("101 =>", predict_prime(101))
print("100 =>", predict_prime(100))

