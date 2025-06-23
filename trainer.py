from typing import Optional
import csv
import matplotlib.pyplot as plt
from predictor import estimatePrice


def loadTrainingData(file_path: str = "data.csv") -> Optional[tuple[list[float], list[float]]]:
    try:
        mileageVec: list[float] = []
        priceVec: list[float] = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                mileageVec.append(float(row[0]))
                priceVec.append(float(row[1]))
        return mileageVec, priceVec
    except OSError:
        print(f"Error: Could not open file '{file_path}'")
        return None
    except ValueError:
        print("Error: File contents are not valid numbers")
        return None


def normalize_data(data: list[float]):
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val

    if range_val == 0:
        return [0.0] * len(data), min_val, 1.0
    normalized = [(x - min_val) / range_val for x in data]
    return normalized, min_val, range_val


def mse(mileageVec, priceVec, t0, t1):
    predictionVec = [estimatePrice(mile, t0, t1) for mile in mileageVec]
    return (1/len(mileageVec)) * sum((prediction - price) ** 2 for prediction, price in zip(predictionVec, priceVec))


def trainModel(mileageVec: list[float], priceVec: list[float], numIterations: int) -> (
        tuple[float, float]):
    t0 = 0.0  # weight
    t1 = 0.0  # bias
    m = len(mileageVec)

    mileageVec, mileage_min, mileage_range = normalize_data(mileageVec)
    priceVec, price_min, price_range = normalize_data(priceVec)

    for i in range(numIterations):
        predictionsVec = [estimatePrice(mileage, t0, t1)
                          for mileage in mileageVec]
        errorsVec = [prediciton - price for prediciton,
                     price in zip(predictionsVec, priceVec)]

        t0 -= learningRate * (1/m) * sum(errorsVec)
        t1 -= learningRate * \
            (1/m) * sum(error * mileage for error,
                        mileage in zip(errorsVec, mileageVec))

    final_t0 = price_min + price_range * t0 - \
        price_range * t1 * mileage_min / mileage_range
    final_t1 = price_range * t1 / mileage_range

    return final_t0, final_t1


def saveWeightsToFile(file_path: str, weights: tuple[float, float]) -> None:
    try:
        with open(file_path, "w") as f:
            f.write(f"{weights[0]}\n{weights[1]}\n")
        print(f"Weights saved to {file_path}")
    except OSError:
        print(f"Error: Could not save weights to '{file_path}'")


def plotData(mileageVec: list[float], priceVec: list[float], weights: Optional[tuple[float, float]] = None) -> None:
    plt.scatter(mileageVec, priceVec, label="Training data", color="blue")

    if weights is not None:
        x_vals = mileageVec
        y_vals = [estimatePrice(x, weights[0], weights[1]) for x in x_vals]
        plt.plot(x_vals, y_vals, color="red", label="Regression line")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Price vs Mileage")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Let's train the model to predict the right price")
    print()

    data = loadTrainingData()
    if data is None:
        exit(1)

    if len(data[0]) <= 0 or len(data[1]) <= 0:
        print("Data length is not greater than 0")
        exit(1)

    mileageVec = data[0]
    priceVec = data[1]

    # Plot data
    # plotData(data)

    learningRate = 0.1
    numIterations = 2000

    # Train modle
    weights = trainModel(mileageVec, priceVec, numIterations)
    print(f"Trained weights: t0 = {weights[0]:.4f}, t1 = {weights[1]:.4f}")

    # Save to file
    saveWeightsToFile("weights.txt", weights)

    # Plot result
    plotData(mileageVec, priceVec, weights)
