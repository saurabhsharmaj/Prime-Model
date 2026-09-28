# -------------------------
# TRAINING DATA
# -------------------------

training_data = [
    (1, "NOT PRIME"),
    (2, "PRIME"),
    (3, "PRIME"),
    (4, "NOT PRIME"),
    (5, "PRIME"),
    (6, "NOT PRIME"),
    (7, "PRIME"),
    (8, "NOT PRIME"),
    (9, "NOT PRIME"),
    (10, "NOT PRIME"),
    (11, "PRIME"),
    (12, "NOT PRIME"),
    (13, "PRIME")
]


# -------------------------
# TEST DATA
# -------------------------

test_data = [
    (17, "PRIME"),
    (19, "PRIME")
]


# -------------------------
# Prepare TRAINING data
# -------------------------

X_train = []
y_train = []

for number, label in training_data:
    X_train.append([number])
    y_train.append(label)


# -------------------------
# Prepare TEST data
# -------------------------

X_test = []
y_test = []

for number, label in test_data:
    X_test.append([number])
    y_test.append(label)


# -------------------------
# Print data
# -------------------------

print("X_train =", X_train)
print("y_train =", y_train)

print("X_test =", X_test)
print("y_test =", y_test)


# -------------------------
# Create model
# -------------------------

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()


# -------------------------
# TRAIN MODEL
# -------------------------

model.fit(X_train, y_train)

print("Model trained!")


# -------------------------
# TEST MODEL
# -------------------------

predictions = model.predict(X_test)

print("Predictions =", predictions)

# -------------------------
# Export MODEL
# -------------------------
import joblib

joblib.dump(model, "prime_model.pkl")

print("Model saved!")