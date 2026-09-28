# -------------------------
# MATHEMATICAL MODEL
# -------------------------

class PrimeModel:

    def is_prime(self, number):

        if number < 2:
            return False

        for i in range(2, int(number ** 0.5) + 1):

            if number % i == 0:
                return False

        return True

    def predict(self, number):

        if self.is_prime(number):
            return "PRIME"

        return "NOT PRIME"