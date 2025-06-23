import numpy as np
import matplotlib.pyplot as plt
from predictor import estimatePrice
from typing import Optional


def loadTrainingData(file_path: str = "data.csv") -> Optional[np.ndarray]:
    try:
        # Assumes header: km,price
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        if data.shape[1] != 2:
            print("Error: data must have two columns: mileage (km) and price")
            return None
        return data
    except OSError:
        print(f"Error: Could not open file '{file_path}'")
        return None
    except ValueError:
        print("Error: File contents are not valid numbers")
        return None


def trainModel(data: np.ndarray, learningRate: float, numIterations: int) -> (
        tuple[float, float]):
    t0 = 0.0 # weight
    t1 = 0.0 # bias
    m = len(data)

    mileageVec = data[:, 0] * 0.0001
    priceVec = data[:, 1] * 0.001

    print("millage:", mileageVec)
    print("price:", priceVec)

    print("price min", np.min(priceVec), "max", np.max(priceVec))
    print("milage min", np.min(mileageVec), "max", np.max(mileageVec))

    for i in range(5):
        predictionsVec = estimatePrice(mileageVec, t0, t1)
        print("predictions:", predictionsVec)

        errorsVec = predictionsVec - priceVec
        print("error:", errorsVec)

        t0 -= learningRate * (1/m) * np.sum(errorsVec)
        t1 += learningRate * (1/m) * np.sum(errorsVec * mileageVec)

        print("scalled errros:", errorsVec * mileageVec)
        print(t0, t1)

    return t0, t1

    for i in range(numIterations):
        predictionsVec = estimatePrice(mileageVec, t0, t1)
        errorsVec = predictionsVec - priceVec

        gradient_t0 = (1 / m) * np.sum(errorsVec)
        gradient_t1 = (1 / m) * np.sum(errorsVec * mileageVec)

        t0 -= learningRate * gradient_t0
        t1 -= learningRate * gradient_t1

        if not np.isfinite(t0) or not np.isfinite(t1):
            print(f"Warning: Non-finite weights at iteration {i}, early exit")
            break

    return t0, t1


def saveWeightsToFile(file_path: str, weights: tuple[float, float]) -> None:
    try:
        with open(file_path, "w") as f:
            f.write(f"{weights[0]}\n{weights[1]}\n")
        print(f"Weights saved to {file_path}")
    except OSError:
        print(f"Error: Could not save weights to '{file_path}'")


def plotData(data: np.ndarray, weights: Optional[tuple[float, float]] = None) -> None:
    plt.scatter(data[:, 0], data[:, 1], label="Training data", color="blue")

    if weights is not None:
        x_vals = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y_vals = estimatePrice(x_vals, weights[0], weights[1])
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

    # Plot data
    #plotData(data)

    learningRate = 0.0001
    numIterations = 1000

    # Train modle
    weights = trainModel(data, learningRate, numIterations)
    print(f"Trained weights: t0 = {weights[0]:.4f}, t1 = {weights[1]:.8f}")

    # Save to file
    saveWeightsToFile("weights.txt", weights)

    # Plot result
    plotData(data, weights)

