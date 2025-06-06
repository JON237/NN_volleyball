"""Trainiert ein neuronales Netz auf Volleyball-Spieldaten.

Dieses Skript unterstützt Teamnamen als Kategorien, die über eine
Embedding-Schicht abgebildet werden. Das verwendete CSV muss daher
zusätzlich zu den numerischen Statistiken die Spalten ``home_team`` und
``away_team`` enthalten.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_data(pfad: str) -> pd.DataFrame:
    """Lädt die Spieldaten aus einer CSV-Datei."""
    return pd.read_csv(pfad)


def build_model(num_numeric: int, num_teams: int, embed_dim: int = 4) -> tf.keras.Model:
    """Erstellt die TensorFlow-Architektur mit Team-Embeddings."""

    # Eingaben für Heim- und Auswärtsteam sowie die numerischen Merkmale
    home_in = tf.keras.layers.Input(shape=(1,), name="home_team")
    away_in = tf.keras.layers.Input(shape=(1,), name="away_team")
    stats_in = tf.keras.layers.Input(shape=(num_numeric,), name="stats")

    # Jedes Team bekommt eine kurze Vektor-Repräsentation
    emb_home = tf.keras.layers.Embedding(num_teams, embed_dim)(home_in)
    emb_away = tf.keras.layers.Embedding(num_teams, embed_dim)(away_in)

    # Embeddings flach machen und mit den numerischen Eingaben kombinieren
    x = tf.keras.layers.Concatenate()(
        [tf.keras.layers.Flatten()(emb_home), tf.keras.layers.Flatten()(emb_away), stats_in]
    )

    # Drei vollverbundene Schichten zur Verarbeitung der Merkmale
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[home_in, away_in, stats_in], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(df: pd.DataFrame) -> tf.keras.Model:
    """Trainiert das Modell und gibt die Genauigkeit aus."""

    # Diese numerischen Spalten werden als Eingaben verwendet
    numeric_cols = [
        "home_rank",
        "away_rank",
        "home_wins_last_10",
        "away_wins_last_10",
        "home_points_last_match",
        "away_points_last_match",
    ]

    # Teamnamen in fortlaufende IDs übersetzen
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    team_to_idx = {team: idx for idx, team in enumerate(sorted(teams))}
    df["home_idx"] = df["home_team"].map(team_to_idx)
    df["away_idx"] = df["away_team"].map(team_to_idx)

    X = df[["home_idx", "away_idx"] + numeric_cols]
    y = df["result"]

    # Datensatz in Trainings- und Testmenge aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Nur die numerischen Spalten skalieren
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_test_num = scaler.transform(X_test[numeric_cols])

    # Eingaben für das Netz vorbereiten
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

    # Abbruch falls sich die Validierungs-Fehler nicht verbessern
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

    # Genauigkeit auf den Testdaten bestimmen
    loss, accuracy = model.evaluate(test_inputs, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.2f}")
    return model


def main():
    """Einstiegspunkt für den Aufruf über die Kommandozeile."""
    import argparse

    # Optionaler Parameter für einen eigenen Datensatz
    parser = argparse.ArgumentParser(
        description="Trainiert ein Vorhersagemodell für Volleyballspiele"
    )
    parser.add_argument(
        "--data",
        default="volleyball_matches_with_teams.csv",
        help="Pfad zur CSV-Datei mit den Spieldaten",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    train_model(df)


if __name__ == '__main__':
    main()
