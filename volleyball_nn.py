"""Train a neural network on volleyball match data using TensorFlow."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_data(path: str) -> pd.DataFrame:
    """Read volleyball match data from a CSV file."""
    return pd.read_csv(path)


def build_model(input_dim: int) -> tf.keras.Model:
    """Create the TensorFlow neural network model."""
    # Simple feed-forward network using three hidden layers
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    # Compile the model with an Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(df: pd.DataFrame) -> tf.keras.Model:
    """Train the TensorFlow model and print its accuracy."""
    # Separate features and label
    X = df.drop("result", axis=1)
    y = df["result"]

    # Split the data so we can evaluate on unseen matches
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize the input features for faster convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])

    # Stop training early if validation loss doesn't improve
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
    )

    # Evaluate the model on the held out test data
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.2f}")
    return model


def main():
    """Entry point when running the script from the command line."""
    import argparse

    # Allow the user to specify a custom dataset file
    parser = argparse.ArgumentParser(
        description="Train volleyball match outcome predictor"
    )
    parser.add_argument(
        "--data", default="volleyball_matches.csv", help="Path to CSV dataset"
    )
    args = parser.parse_args()

    df = load_data(args.data)
    train_model(df)


if __name__ == '__main__':
    main()
