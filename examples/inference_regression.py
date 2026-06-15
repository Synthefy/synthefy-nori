from synthefy_nori import NoriRegressor


def main():
    X_train = [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    y_train = [0.0, 1.0, 2.0]
    X_test = [[1.5, 0.5]]

    model = NoriRegressor()
    print(model.fit(X_train, y_train).predict(X_test))


if __name__ == "__main__":
    main()
