"""Veracity prediction for claimdecomp.

python3 veracity_prediction.py --test_path path to test file
-- model_path path to model --questions_path path to decomposed questions from claim
-- output_path output/...
"""
import json
import argparse
from code.utils.data_loader import read_json
from typing import Dict, List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from torch import Tensor
from code.utils.load_veracity_predictor import VeracityClassifier
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))

def get_top_k_similar_instances(
    sentence: str, data_emb: Tensor, data: List[Dict],
    k: int, threshold: float
) -> List[Dict]:
    """get top k neighbours for a sentence.

    Args:
        sentence (str): input
        data_emb (Tensor): corpus embeddings
        data (List[Dict]): corpus
        k (int): top_k to return
        threshold (float):

    Returns:
        List[Dict]: list of top_k data points
    """
    sent_emb = model.encode(sentence)
    # data_emb = self.get_embeddings_for_data(transfer_questions)
    print("new_emb", sent_emb.shape, data_emb.shape)
    text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
    results_sims = zip(range(len(text_sims)), text_sims)
    sorted_similarities = sorted(
        results_sims, key=lambda x: x[1], reverse=True)
    print("text_sims", sorted_similarities[:2])
    top_questions = []
    for idx, item in sorted_similarities[:k]:
        if item[0] > threshold:
            top_questions.append(list(data)[idx])
    return top_questions


def get_verification(config):
    """Get veracity predictions."""
    with open(config["bm25_evidence_path"]) as f:
        data = json.load(f)
    facts = read_json(
        config["test_path"]
    )

    decomposed_questions = pd.read_csv(
        config["questions_path"], sep="@"
    )

    model_name = config["model_path"]

    nli_model = VeracityClassifier(
        base_model=config["base_model"], model_name=model_name
    )
    results = []
    matches = 0
    unmatches = 0
    verdicts = {"claim": [], "verdict": []}
    print("Questions:",decomposed_questions)
    for index, fact in enumerate(facts):
        assert decomposed_questions.shape[0] == len(facts)
        assert data[index]["claim"] == fact["claim"]
        if decomposed_questions.iloc[index]["claims"] != fact["claim"]:
            print(
                "not equal", decomposed_questions.iloc[index]["claims"], fact["claim"]
            )

        questions = decomposed_questions.iloc[index]["questions"]
        questions = questions.lower().split("next question:")
        print("questions", questions)
        result = {"evidences": []}
        result["claim"] = fact["claim"]
        top_100_docs = data[index]["docs"]
        print("top_100_docs", len(top_100_docs), len(list(set(top_100_docs))))
        doc_embeddings = model.encode(top_100_docs)
        top_k_docs = []
        for question in questions:
            print("question", question)
            top_1_docs = get_top_k_similar_instances(
                question, doc_embeddings, top_100_docs, 1, 0.5
            )
            top_k_docs.extend(top_1_docs)
        if len(top_k_docs) == 0 and len(questions) > 0:
            top_k_docs = get_top_k_similar_instances(
                questions[0], doc_embeddings, top_100_docs, 1, 0.5
            )
        print(len(top_k_docs), len(list(set(top_k_docs))))
        top_k_docs = list(set(top_k_docs))
        questions = list(set(questions))
        print("top_k_docs", len(top_k_docs))
        verdicts["claim"].append(fact["claim"])
        if len(top_k_docs) > 0:
            for doc in top_k_docs:
                result["evidences"].append(doc)
            input = (
                "[Claim]: "
                + fact["claim"]
                + "[Questions]: "
                + " ".join(questions)
                + "[Evidences]:"
                + " ".join(top_k_docs)
            )
            pred_label, _ = nli_model.predict(input, max_legnth=256)
        elif len(top_k_docs) == 0:
            print("No documents retrieved verifying claim directly")
            pred_label, _ = nli_model.predict(fact["claim"])
        # pred_label = pred_label if abs(probs[1]-probs[0]) > 0.2 else "NONE"
        print("pred_label", pred_label)
        if pred_label == "SUPPORTS":
            verdict = "True"
        elif pred_label == "REFUTES":
            verdict = "False"
        elif pred_label == "CONFLICTING":
            verdict = "Conflicting"

        print("Verdict:", verdict)
        verdicts["verdict"].append(verdict)
        results.append(result)
        verdict_1 = pd.DataFrame(verdicts)
        print(verdict_1)
        output_path = config["output_path"]
        verdict_1.to_csv(f"output/{output_path}.csv", index=False)
        print(f"{fact['claim']}\t{fact['label']}\t{verdict}")
        if verdict == fact["label"]:
            matches += 1
        else:
            unmatches += 1
        print("accuracy", matches / (matches + unmatches))
        with open(f"output/{output_path}.json", "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", type=str,
                        default="data/raw_data/test_claims_quantemp.json",
                        help="Path to the test data")
    parser.add_argument("--bm25_evidence_path", type=str,
                        default="data/bm25_scored_evidence/bm25_top_100_claimdecomp.json",
                        help="Path to the top 100 bm25 docs")
    parser.add_argument("--base_model", type=str,
                        default="roberta-large-mnli", help="Path to the base model")
    parser.add_argument("--model_path", type=str,
                        default="models/finqa_roberta_claimdecomp_early_stop_2/model_weights.zip",
                        help="Path to the tokenizer")
    parser.add_argument("--questions_path", type=str,
                        default="data/decomposed_questions/test/test_claimdecomp.csv",
                        help="Path to the decomposed questions")
    parser.add_argument("--output_path", type=str,
                        default="finqa_roberta_claimdecomp_test",
                        help="Path to the output predictions")
    args = parser.parse_args()

    CONFIG = {
        "bm25_evidence_path": args.bm25_evidence_path,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "test_path": args.test_path,
        "questions_path": args.questions_path,
        "output_path": args.output_path
    }
    get_verification(CONFIG)
