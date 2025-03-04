""" Trains a setfit model for categorizing claims to numerical taxonomy
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from setfit import SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss
import os
from setfit import SetFitTrainer, sample_dataset
from typing import List, Dict, Any


def get_encoded_labels(targets: List) -> Any:
    """encode category labels

    Args:
        targets (List): list of trainign labels

    Returns:
        Any: labelencoder and list of labels
    """
    LE = LabelEncoder()
    train_labels = LE.fit_transform(targets)
    return LE, train_labels

# sentence-transformers/paraphrase-mpnet-base-v2


def setup_train(data_files: Dict, model_name: str, 
                iterations: int, source_column_name: str, target_column_name: str):
    """training script for training the setfit model for categorizing claims

    Args:
        data_files (Dict): has train , val and test paths
        model_name (str): pre-trained model to fine tune for taxonomy 
        categorization
        iterations (int): number of iterations
        source_column_name (str): input field
        target_column_name (str): output field
    """
    raw_datasets = load_dataset("csv", data_files=data_files)
    model = SetFitModel.from_pretrained(model_name)
    train_dataset = sample_dataset(
        raw_datasets["train"], label_column="label", num_samples=20)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=raw_datasets["validation"],
        loss_class=CosineSimilarityLoss,
        num_iterations=iterations,
        column_mapping={source_column_name: "text",
                        target_column_name: "label"},
    )
    trainer.train()
    trainer.evaluate()
    model._save_pretrained(
        str(os.path.join(dir_path, "models/setfit/taxonomy_classifier")))


def create_setfit_data(train: pd.DataFrame, 
                       test: pd.DataFrame, val: pd.DataFrame) -> Dict:
    """creates train test val splits for training the setfit classifier

    Args:
        train (pd.DataFrame): trainign data with category labels
        test (pd.DataFrame): test data with category labels
        val (pd.DataFrame): validation data with category labels

    Returns:
        Dict: dictionary of paths for train,val and test data
    """
    LE, labels = get_encoded_labels(train["taxonomy_labels"].values)
    train["label"] = labels

    test["label"] = LE.transform(test["taxonomy_labels"].values)
    val["label"] = LE.transform(val["taxonomy_labels"].values)
    train.to_csv(
        "data/taxonomy_classifier_data/train_taxonomy.csv", index=False)
    val.to_csv("data/taxonomy_classifier_data/val_taxonomy.csv", index=False)
    test.to_csv("data/taxonomy_classifier_data/test_taxonomy.csv", index=False)
    dir_path = os.path.dirname(os.path.realpath(os.getcwd()))

    data_files = {}
    data_files["train"] = os.path.join(
        dir_path, "factiverse/data/taxonomy_classifier_data/train_taxonomy.csv")
    data_files["validation"] = os.path.join(
        dir_path, "factiverse/data/taxonomy_classifier_data/val_taxonomy.csv")
    return data_files


data = pd.read_csv(
    "/Users/venktesh_1/Downloads/factiverse/complex_num_facts_categorized_subset.csv", sep="@")
train, val = train_test_split(data, test_size=0.2, random_state=42)
train, test = train_test_split(train, test_size=0.2, random_state=42)
data_files = create_setfit_data(train, test, val)
setup_train(data_files, "sentence-transformers/paraphrase-mpnet-base-v2",
            20, "claim", "label")
