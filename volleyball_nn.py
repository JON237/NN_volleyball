"""Train a neural network on volleyball match data using TensorFlow."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_data(path: str) -> pd.DataFrame:
    """Load volleyball match data from a CSV file."""
    return pd.read_csv(path)


def build_model(input_dim: int) -> tf.keras.Model:
    """Create the TensorFlow neural network model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(df: pd.DataFrame) -> tf.keras.Model:
    """Train the TensorFlow model and print its accuracy."""
    X = df.drop("result", axis=1)
    y = df["result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
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
