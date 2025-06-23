
def estimatePrice(mileage: float, t0: float, t1: float) -> float:
    '''price = t0 + (t1 * mileage)'''
    return t0 + (t1 * mileage)


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_weights(file_path="weights.txt") -> tuple[float | None, float | None]:
    try:
        with open(file_path, "r") as file:
            lines: list[str] = file.readlines()
            if len(lines) < 2:
                print(
                    f"Error: {file_path} should contain two lines (t0 and t1)")
                return None, None

            t0_str: str = lines[0].strip()
            t1_str: str = lines[1].strip()

            if not is_number(t0_str) or not is_number(t1_str):
                print("Error: t0 and t1 must be numbers.")
                return None, None

            return float(t0_str), float(t1_str)
    except OSError:
        print("Error: weights.txt is inaccessible")
        return None, None


if __name__ == "__main__":
    print("Let's predict the price of a car!")
    print()

    t0: float | None
    t1: float | None
    t0, t1 = load_weights()
    if t0 is None or t1 is None:
        exit(1)

    mileage_input: str = input("Enter the car's mileage: ").strip()
    if not is_number(mileage_input):
        print("Error: Mileage must be a number")
        exit(1)

    mileage: float = float(mileage_input)
    estimated_price: float = estimatePrice(mileage, t0, t1)
    print(f"Estimated car at {mileage}km: {estimated_price:.2f}")
