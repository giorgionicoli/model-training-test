import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime
    import json
    import pickle

    import polars as pl
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from tqdm.notebook import tqdm
    import optuna
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    return (
        MultiLabelBinarizer,
        MultilabelStratifiedKFold,
        TfidfVectorizer,
        cross_val_score,
        datetime,
        np,
        optuna,
        pickle,
        pl,
        xgb,
    )


@app.cell
def _(
    MultiLabelBinarizer,
    MultilabelStratifiedKFold,
    TfidfVectorizer,
    cross_val_score,
    np,
    optuna,
    pickle,
    pl,
    xgb,
):
    def load_dataset(filename: str) -> pl.DataFrame:
        return pl.read_parquet(filename)


    def load_embedding_model(filename: str) -> TfidfVectorizer:
        with open(filename, "rb") as fh:
            return pickle.load(fh)


    def store_artifact_as_pickle(artifact, filename) -> None:
        with open(filename, "wb") as _fh:
            pickle.dump(artifact, _fh)


    def prepare_dataset_train_test_splits(dataset: pl.DataFrame) -> tuple[
        list[str], list[str], list[list[int]], list[list[int]]
    ]:
        sample_identifiers = dataset["alert_id"].to_list()
        X = dataset["text"].to_list()
        Y = dataset["labels"].to_list()
        mlb = MultiLabelBinarizer()
        Y_binarized = mlb.fit_transform(Y)

        mlskf = MultilabelStratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )
        splits = mlskf.split(sample_identifiers, Y_binarized)
        train_idx, test_idx = next(splits)

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        Y_train = [Y[i] for i in train_idx]
        Y_test = [Y[i] for i in test_idx]

        return X_train, X_test, Y_train, Y_test


    def optimize_hyperparameters(
        X_emb: np.ndarray, Y_bin: np.ndarray, n_trials: int = 30
    ) -> dict:
        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "eval_metric": "logloss",
                "tree_method": "hist",
            }

            model = xgb.XGBClassifier(**params)
            score = cross_val_score(
                model, X_emb, Y_bin, cv=5, scoring="f1_micro"
            ).mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params


    def train_classifier(X_train: list[str], Y_train: list[list[int]], embedder: TfidfVectorizer) -> xgb.XGBClassifier:
        mlb = MultiLabelBinarizer()
        Y_train_binarized = mlb.fit_transform(Y_train)
        X_train_embeddings = embedder.transform(X_train)
        best_params = optimize_hyperparameters(
            X_train_embeddings, Y_train_binarized
        )
        clf_head = xgb.XGBClassifier(**best_params)
        clf_head.fit(X_train_embeddings, Y_train_binarized)

        return clf_head
    return (
        load_dataset,
        load_embedding_model,
        prepare_dataset_train_test_splits,
        store_artifact_as_pickle,
        train_classifier,
    )


@app.cell
def _(
    load_dataset,
    load_embedding_model,
    prepare_dataset_train_test_splits,
    store_artifact_as_pickle,
    train_classifier,
):
    def main(
        *,
        dataset_filename: str,
        embedding_model_filename: str,
        output_filename: str,
    ) -> None:
        dataset = load_dataset(dataset_filename)
        embedder = load_embedding_model(embedding_model_filename)

        X_train, X_test, Y_train, Y_test = prepare_dataset_train_test_splits(
            dataset
        )

        classifier = train_classifier(X_train, Y_train, embedder)

        print("Model trained successfully!")
        store_artifact_as_pickle(
            {
                "embedder": embedder,
                "classifier": classifier,
            },
            output_filename,
        )
        print("Model stored successfully at:", output_filename)
    return (main,)


@app.cell
def _(datetime, main, time_str):
    _time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    main(
        dataset_filename="data/<dataset-filename-placeholder>.parquet",
        embedding_model_filename="models/<embedding-model-filename-placeholder>.pkl",
        output_filename=f"models/{time_str}_<output-model-filename-placeholder>.pkl",
    )
    return


if __name__ == "__main__":
    app.run()
