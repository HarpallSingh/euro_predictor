import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV data
def load_data(filepath):
    return pd.read_csv(filepath)

# Filter the data
def filter_data(data):
    historical_matches = data[data['match_status'] != 'not_started'].copy()
    upcoming_matches = data[data['match_status'] == 'not_started'].copy()
    return historical_matches, upcoming_matches

# Calculate win rates for home and away teams
def team_stats(matches, team, location):
    if location == 'home':
        wins = matches[(matches['home_name'] == team) & (matches['home_score'] > matches['away_score'])].shape[0]
        total = matches[matches['home_name'] == team].shape[0]
    else:
        wins = matches[(matches['away_name'] == team) & (matches['away_score'] > matches['home_score'])].shape[0]
        total = matches[matches['away_name'] == team].shape[0]
    return wins / total if total > 0 else 0

# Add features to historical matches
def add_features(historical_matches):
    historical_matches['home_win_rate'] = historical_matches['home_name'].apply(lambda x: team_stats(historical_matches, x, 'home'))
    historical_matches['away_win_rate'] = historical_matches['away_name'].apply(lambda x: team_stats(historical_matches, x, 'away'))
    return historical_matches

# Prepare training data
def prepare_training_data(historical_matches):
    X = historical_matches[['home_win_rate', 'away_win_rate']]
    y_home = historical_matches['home_score']
    y_away = historical_matches['away_score']
    return train_test_split(X, y_home, test_size=0.2, random_state=42), train_test_split(X, y_away, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')

# Add features to upcoming matches
def add_upcoming_features(upcoming_matches, historical_matches):
    upcoming_matches['home_win_rate'] = upcoming_matches['home_name'].apply(lambda x: team_stats(historical_matches, x, 'home'))
    upcoming_matches['away_win_rate'] = upcoming_matches['away_name'].apply(lambda x: team_stats(historical_matches, x, 'away'))
    return upcoming_matches[['home_win_rate', 'away_win_rate']]

# Predict scores for upcoming matches
def predict_scores(model_home, model_away, X_upcoming):
    predicted_home_scores = model_home.predict(X_upcoming)
    predicted_away_scores = model_away.predict(X_upcoming)
    return predicted_home_scores, predicted_away_scores

def main():
    # Load data
    filepath = r'EM 2024 pairings (1).csv'
    data = load_data(filepath)

    # Filter data
    historical_matches, upcoming_matches = filter_data(data)

    # Debugging: Print the shape of the datasets
    print(f'Historical matches shape: {historical_matches.shape}')
    print(f'Upcoming matches shape: {upcoming_matches.shape}')

    # Add features to historical matches
    historical_matches = add_features(historical_matches)

    # Prepare training data
    (X_train_home, X_test_home, y_train_home, y_test_home), (X_train_away, X_test_away, y_train_away, y_test_away) = prepare_training_data(historical_matches)

    # Train the models
    model_home = train_model(X_train_home, y_train_home)
    model_away = train_model(X_train_away, y_train_away)

    # Evaluate the models
    evaluate_model(model_home, X_test_home, y_test_home)
    evaluate_model(model_away, X_test_away, y_test_away)

    # Add features to upcoming matches
    X_upcoming = add_upcoming_features(upcoming_matches, historical_matches)

    # Predict scores for upcoming matches
    upcoming_matches['predicted_home_score'], upcoming_matches['predicted_away_score'] = predict_scores(model_home, model_away, X_upcoming)

    # Debugging: Print predictions for all upcoming matches
    print(upcoming_matches[['home_name', 'away_name', 'predicted_home_score', 'predicted_away_score']])

    # Print the predicted outcome for England vs Spain
    match_england_spain = upcoming_matches[
        (upcoming_matches['home_name'] == 'England') & (upcoming_matches['away_name'] == 'Spain')
    ]
    if not match_england_spain.empty:
        print("Prediction for England vs Spain:")
        print(f"Home Score: {match_england_spain['predicted_home_score'].values[0]}")
        print(f"Away Score: {match_england_spain['predicted_away_score'].values[0]}")
    else:
        print("Match England vs Spain not found in upcoming matches.")

if __name__ == "__main__":
    main()
