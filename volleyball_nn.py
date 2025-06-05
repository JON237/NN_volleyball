import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


def load_data(path: str) -> pd.DataFrame:
    """Load volleyball match data from a CSV file."""
    return pd.read_csv(path)


def train_model(df: pd.DataFrame) -> MLPClassifier:
    """Train a neural network to predict match results."""
    X = df.drop('result', axis=1)
    y = df['result']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train volleyball match outcome predictor"
    )
    parser.add_argument(
        '--data', default='volleyball_matches.csv', help='Path to CSV dataset'
    )
    args = parser.parse_args()
    df = load_data(args.data)
    train_model(df)


if __name__ == '__main__':
    main()
