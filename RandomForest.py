import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, classification_report

if __name__ == '__main__':
    little_data_df = pd.read_csv("little_data.csv")
    big_data_df = pd.read_csv("big_data.csv").set_index('player')

    features = [
        'aces_and_unret_percentage', 'df_percent', 'first_won_percent',
        'forced_per_point', 'unforced_per_point', 'winners_per_point',
        'net_frequency', 'net_points_won_ratio', 'net_winner_rate',
        'net_unforced_rate', 'net_forced_rate',
        'return_points_won_percent'
    ]
    X = big_data_df[features].copy()

    features_df = big_data_df.reset_index()

    scaler = RobustScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)


    # create rule-based initial labels for bootstrapping
    y_server = ((big_data_df['aces_and_unret_percentage'] > big_data_df['aces_and_unret_percentage'].quantile(0.75)) &
                (big_data_df['first_won_percent'] > 0.72)).astype(int)
    y_aggr = ((big_data_df['winners_per_point'] > big_data_df['winners_per_point'].quantile(0.7)) &
              (big_data_df['unforced_per_point'] < big_data_df['unforced_per_point'].quantile(0.7))).astype(int)
    y_net = ((big_data_df['net_frequency'] > big_data_df['net_frequency'].quantile(0.75)) &
             (big_data_df['net_points_won_ratio'] > big_data_df['net_points_won_ratio'].quantile(.6))).astype(int)
    y_def = (big_data_df['return_points_won_percent'] > big_data_df['return_points_won_percent'].quantile(0.7)).astype(int)
    print("75th percentile of aces and unret percentage: ", big_data_df['aces_and_unret_percentage'].quantile(0.75))
    label_sets = {
        'server': y_server,
        'aggressive': y_aggr,
        'net': y_net,
        'defensive': y_def
    }

    models = {}
    calibrated = {}
    thresholds = {}

    for name, y in label_sets.items():
        print(y)
        X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
        tree.fit(X_train, y_train)
        # calibrate probabilities
        calib = CalibratedClassifierCV(tree, method='sigmoid', cv='prefit')
        calib.fit(X_val, y_val)
        prob_val = calib.predict_proba(X_val)[:, 1]
        prec, rec, thresh = precision_recall_curve(y_val, prob_val)
        print(prec)
        # choose threshold at max F1 or a point with desired precision/recall tradeoff
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        # best_idx = f1_scores.argmax()
        best_idx = prec.argmax()
        best_thresh = thresh[best_idx] if best_idx < len(thresh) else 0.5
        # store
        models[name] = tree
        calibrated[name] = calib
        thresholds[name] = best_thresh

        print(f"{name} threshold {best_thresh:.3f}")
        print(export_text(tree, feature_names=list(Xs.columns)))
        print(classification_report(y_val, (prob_val >= best_thresh).astype(int)))

    def predict_archetypes(player_row):
        x = scaler.transform(player_row[features].values.reshape(1, -1))
        preds = {}
        for name, calib in calibrated.items():
            p = calib.predict_proba(x)[0, 1]
            preds[name] = (p, p >= thresholds[name])
        return preds


    print("Player: ", little_data_df.iloc[386, 0])
    print(predict_archetypes(big_data_df.iloc[386]))


    print("hi")
