from predictor import loadWeights, estimatePrice
from trainer import loadTrainingData


def meanSquaredError(mileageVec: list[float], priceVec: list[float], t0: float, t1: float) -> float:
    predictionVec = [estimatePrice(mile, t0, t1) for mile in mileageVec]
    return (1/len(mileageVec)) * sum((prediction - price) ** 2 for prediction, price in zip(predictionVec, priceVec))


if __name__ == "__main__":
    print("Let's verify the precision of the model")
    print()

    t0, t1 = loadWeights()
    if t0 is None or t1 is None:
        exit(1)

    mileageVec, priceVec = loadTrainingData()
    if mileageVec is None or priceVec is None:
        exit(1)

    if len(mileageVec) <= 0 or len(priceVec) <= 0:
        print("Error: Data length is not greater than 0")
        exit(1)

    mse = meanSquaredError(mileageVec, priceVec, t0, t1)
    rmse = mse ** 0.5
    print(f"Root mean squared error is : €{rmse:.2f}")
    print("On average the model is off by this amount for the car price.")
