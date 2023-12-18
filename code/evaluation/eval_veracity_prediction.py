import pandas as pd
import evaluate
import json
import argparse
from typing import List,Dict
from sklearn.metrics import classification_report, confusion_matrix,f1_score

f1_metric = evaluate.load("f1")
def get_accuracy(predictions, gold_labels):
    matches=0
    unmatches=0
    for index,prediction in enumerate(predictions):
        try:
            prediction = prediction.split("Verdict:")[1]
        except:
            unmatches+=1
        prediction = prediction.replace(".", "")
        if prediction.lower().strip() == gold_labels[index]["label"].lower():
            matches += 1
        else:
            unmatches += 1
    print("Accuracy:", matches / (matches + unmatches))

def print_evaluation_results(predictions, gt_labels, num_of_classes=3):
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

def get_direct_accuracy(predictions, gold_labels):
    matches=0
    unmatches=0
    for index,prediction in enumerate(predictions):
        #print(prediction.lower().strip(),gold_labels[index]["label"].lower())
        if prediction.lower().strip() == gold_labels[index]["label"].lower():
            matches += 1
        else:
            unmatches += 1
    print("Accuracy:", matches / (matches + unmatches))

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
    parser.add_argument("--output_path", type=str,
                        default="output/finqa_roberta_claimdecomp_test.csv",
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
        "output_path": args.output_path,
        "category": args.category,
        "category_wise": args.category_wise
    }

    ground_truth_data = read_claims_data(CONFIG["test_path"], 
                                         CONFIG["category"],
                                         CONFIG["category_wise"])
    print("ground_truth_data",len(ground_truth_data))
    test_set = [claim["claim"].lower().strip() for claim in ground_truth_data]

    claimdecomp_unified = pd.read_csv(CONFIG["output_path"])
    print("claimdecomp_unified",len(claimdecomp_unified))
    claimdecomp_unified = claimdecomp_unified[claimdecomp_unified["claim"].str.lower().str.strip().isin(test_set)]
    print("claimdecomp_unified",len(claimdecomp_unified))


ground_truth = [dat["label"] for dat in ground_truth_data][:len(claimdecomp_unified)]
#predictions = ["False" for dat in numdecomp_oracle["verdict"].values]
print_evaluation_results(claimdecomp_unified["verdict"].values,ground_truth,3)