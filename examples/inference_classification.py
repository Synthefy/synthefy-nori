from synthefy_tabular import SynthefyTabularClassifier


def main():
    X_train = [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.0]]
    y_train = ["low", "low", "high", "high"]
    X_test = [[1.5, 0.5]]

    model = SynthefyTabularClassifier()
    print(model.fit(X_train, y_train).predict(X_test))


if __name__ == "__main__":
    main()
