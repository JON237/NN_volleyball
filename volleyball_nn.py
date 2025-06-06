"""Train a neural network on volleyball match data using TensorFlow.

This version supports categorical team names which are fed through an
embedding layer.  The CSV file therefore needs the columns
``home_team`` and ``away_team`` in addition to the numeric statistics.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_data(path: str) -> pd.DataFrame:
    """Read volleyball match data from a CSV file."""
    return pd.read_csv(path)


def build_model(num_numeric: int, num_teams: int, embed_dim: int = 4) -> tf.keras.Model:
    """Create the TensorFlow neural network model with team embeddings."""
    # Inputs for the home/away team identifiers and the numeric stats
    home_in = tf.keras.layers.Input(shape=(1,), name="home_team")
    away_in = tf.keras.layers.Input(shape=(1,), name="away_team")
    stats_in = tf.keras.layers.Input(shape=(num_numeric,), name="stats")

    # Look up embeddings for both teams
    emb_home = tf.keras.layers.Embedding(num_teams, embed_dim)(home_in)
    emb_away = tf.keras.layers.Embedding(num_teams, embed_dim)(away_in)

    # Flatten the embeddings and concatenate with the numeric inputs
    x = tf.keras.layers.Concatenate()(\
        [tf.keras.layers.Flatten()(emb_home), tf.keras.layers.Flatten()(emb_away), stats_in]
    )

    # Feed-forward layers
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[home_in, away_in, stats_in], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(df: pd.DataFrame) -> tf.keras.Model:
    """Train the TensorFlow model and print its accuracy."""

    numeric_cols = [
        "home_rank",
        "away_rank",
        "home_wins_last_10",
        "away_wins_last_10",
        "home_points_last_match",
        "away_points_last_match",
    ]

    # Convert team names to integer IDs starting at 0
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    team_to_idx = {team: idx for idx, team in enumerate(sorted(teams))}
    df["home_idx"] = df["home_team"].map(team_to_idx)
    df["away_idx"] = df["away_team"].map(team_to_idx)

    X = df[["home_idx", "away_idx"] + numeric_cols]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale only the numeric columns
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_test_num = scaler.transform(X_test[numeric_cols])

    # Prepare model inputs as lists
    train_inputs = [
        X_train["home_idx"].values,
        X_train["away_idx"].values,
        X_train_num,
    ]
    test_inputs = [
        X_test["home_idx"].values,
        X_test["away_idx"].values,
        X_test_num,
    ]

    model = build_model(len(numeric_cols), len(team_to_idx))

    # Stop training early if validation loss doesn't improve
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_inputs,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
    )

    # Evaluate the model on the held out test data
    loss, accuracy = model.evaluate(test_inputs, y_test, verbose=0)
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
        "--data",
        default="volleyball_matches_with_teams.csv",
        help="Path to CSV dataset",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    train_model(df)


if __name__ == '__main__':
    main()
