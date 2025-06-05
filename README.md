# NN_volleyball

This repository contains a minimal example of training a neural network to predict the outcome of volleyball matches.

## Dataset

The training data is expected as a CSV file where each row represents one past match. An example dataset `volleyball_matches.csv` is included. The columns used are:

- `home_rank` – rank of the home team
- `away_rank` – rank of the away team
- `home_wins_last_10` – wins by the home team in the last 10 matches
- `away_wins_last_10` – wins by the away team in the last 10 matches
- `home_points_last_match` – points scored by the home team in the previous match
- `away_points_last_match` – points scored by the away team in the previous match
- `result` – 1 if the home team won, 0 otherwise

## Training

Install the dependencies and run `volleyball_nn.py`:

```bash
pip install tensorflow scikit-learn pandas streamlit
python volleyball_nn.py --data volleyball_matches.csv
```

The script splits the data into training and test sets, scales the features and trains a TensorFlow network with layers 64-32-16 using ReLU activations and a sigmoid output. The test accuracy is printed after training.

## Interactive Demo

To experiment with your own inputs, launch the Streamlit application:

```bash
streamlit run streamlit_app.py --server.headless true
```

The web interface lets you enter team statistics and returns which team the model expects to win along with the predicted probability.
