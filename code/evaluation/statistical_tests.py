import json
from sklearn.model_selection import KFold
from scipy import stats
import pandas as pd
import random
import argparse
import evaluate
from typing import List,Dict

from sklearn.metrics import classification_report, confusion_matrix,f1_score

f1_metric = evaluate.load("f1")

def get_f1_scores(predictions, gt_labels, num_of_classes=3):
    if num_of_classes == 3:
        target_names = ['False', 'True','Conflicting']
        label_map = {'False': 0, 'True': 1, 'Conflicting': 2}
        label_map_1 = {'False': 0, 'True': 1, 'Conflicting': 2}
        labels = [label_map[e] for e in gt_labels]
        predictions = [label_map[str(e)] for e in predictions]
        print(classification_report(labels, predictions, target_names=target_names, digits=4))
        print(confusion_matrix(labels, predictions))
        print(f1_metric.compute(references=labels,predictions=predictions,average="weighted"))
        print(f1_metric.compute(references=labels,predictions=predictions,average="macro"))
        print(f1_metric.compute(references=labels,predictions=predictions,average="micro"))

        print()
    elif num_of_classes == 2:
        target_names = ['True', 'False']
        label_map = {'refutes': 0, 'supports': 1}
        labels = [label_map[e] for e in gt_labels]
        predictions = [label_map[e] for e in predictions]
        print(classification_report(labels, predictions, target_names=target_names, digits=4))
        print(confusion_matrix(labels, predictions))
        print()
    return (f1_metric.compute(references=labels,predictions=predictions,average="macro")["f1"],
    f1_metric.compute(references=labels,predictions=predictions,average="weighted")["f1"])


def t_test(approach_1,baseline, original_data, labels):
    macro_ours = []
    macro_baseline_scores = []
    weighted_ours = []
    weighted_baseline_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for index_fold in kf.split(original_data):
        labels_split = []
        for idx in index_fold[1]:
            labels_split.append(labels[idx])
        our_f1_scores = get_f1_scores(approach_1.iloc[index_fold[1],:]["verdict"].values, labels_split)
        baseline_scores = get_f1_scores(baseline.iloc[index_fold[1],:]["verdict"].values, labels_split)
        macro_ours.append(our_f1_scores[0])
        macro_baseline_scores.append(baseline_scores[0])
        weighted_ours.append(our_f1_scores[1])
        weighted_baseline_scores.append(baseline_scores[1])
    print(macro_ours, macro_baseline_scores)
    print("macro statistical significance metrics", stats.ttest_rel(macro_ours, macro_baseline_scores))
    print("weighted statistical significance metrics", stats.ttest_rel(weighted_ours, weighted_baseline_scores))

def read_claims_data(test_path: str, category: str, category_wise: bool) -> List[Dict]:
    """_summary_

    Args:
        category (str): the class of claims to filter out
        category_wise (bool): shoudl the filtering happen?

    Returns:
        List[Dict]: list of fact objects
    """  
    with open(test_path) as f:
        data = json.load(f)
    categorized_data = []
    for dat in data:
    #
     #
        if category_wise:
            if dat["taxonomy_label"] == category:
                categorized_data.append(dat)
        else:
            categorized_data.append(dat)
    return categorized_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", type=str,
                        default="data/raw_data/test_claims_quantemp.json",
                        help="Path to the test data")
    parser.add_argument("--output_path_1", type=str,
                        default="output/unified_claimdecomp_final.csv",
                        help="Path to the output predictions")
    parser.add_argument("--output_path_2", type=str,
                        default="output/claim_only_numdecomp_final_5.csv",
                        help="Path to the output predictions")
    parser.add_argument("--category", type=str,
                        default="statistical",
                        help="The class of claims to report results for")
    parser.add_argument("--category_wise", type=bool,
                        default=False,
                        help="should classwise filtering be applied")
    args = parser.parse_args()

    CONFIG = {
        "test_path": args.test_path,
        "output_path_1": args.output_path_1,
        "output_path_2": args.output_path_2,
        "category": args.category,
        "category_wise": args.category_wise
    }
    ground_truth_data = read_claims_data(CONFIG["test_path"], 
                                         CONFIG["category"],
                                         CONFIG["category_wise"])
    print("ground_truth_data",len(ground_truth_data))
    test_set = [claim["claim"].lower().strip() for claim in ground_truth_data]

    approach_1 = pd.read_csv(CONFIG["output_path_1"])
    print("approach_1",len(approach_1))
    approach_1 = approach_1[approach_1["claim"].str.lower().str.strip().isin(test_set)]
    print("approach_1",len(approach_1))


    approach_2 = pd.read_csv(CONFIG["output_path_2"])
    print("approach_2",len(approach_2))
    approach_2 = approach_2[approach_2["claim"].str.lower().str.strip().isin(test_set)]
    print("approach_2",len(approach_2))
    labels = [dat["label"] for dat in ground_truth_data]
    t_test(approach_1, approach_2, ground_truth_data, labels)