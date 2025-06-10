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
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    return (
        MultiLabelBinarizer,
        MultilabelStratifiedKFold,
        TfidfVectorizer,
        datetime,
        pickle,
        pl,
        xgb,
    )


@app.cell
def _(
    MultiLabelBinarizer,
    MultilabelStratifiedKFold,
    TfidfVectorizer,
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


    def train_classifier(X_train: list[str], Y_train: list[list[int]], embedder: TfidfVectorizer) -> xgb.XGBClassifier:
        mlb = MultiLabelBinarizer()
        Y_train_binarized = mlb.fit_transform(Y_train)
        X_train_embeddings = embedder.transform(X_train)

        clf_head = xgb.XGBClassifier()
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
