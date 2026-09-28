import joblib

model = joblib.load("prime_model.pkl")

number = int(input("Enter a number: "))

result = model.predict([[number]])

print(number, "=>", result[0])