from datasets import load_metric
from statistics import mean,median,stdev
import pandas as pd
from utils.fluency import Fluency
import json
from utils.diversity import diversity,diversity_single

bertscore_metric = load_metric('bertscore')

def get_bert_scores(claim,question):
    bert_scores = bertscore_metric.compute(predictions=question, references=claim, lang="en",model_type="distilbert-base-uncased")

    return bert_scores


def get_scores(dataframe):
    questions_list = []
    claims = []
    for index, claim in dataframe.iterrows():
        questions = dataframe.iloc[index]["questions"].lower().split("next question:")[1:]
        for question in questions:
            questions_list.append(question)
            claims.append(claim["claims"])
    bert_scores = get_bert_scores(claims, questions_list)
    return bert_scores

if __name__ =="__main__":
    questions_list = []
    claims = []
    fluency = Fluency(model_name="gpt2")
    numdecomp_claims_with_questions = pd.read_csv("output/decomposed_claim_questions_final.csv",sep="@")
    claimdecomp_claims_with_questions = pd.read_csv("output/decomposed_claim_yes_no_questions.csv",sep="@")

    claims_list = [dat["claims"].lower().strip() for index, dat in numdecomp_claims_with_questions.iterrows()]
    programs = []
    with open("/Users/venktesh_1/Downloads/ProgramFC/results/programs/NumDecomp_N=1_gpt-3.5-turbo_programs.json") as f:

        programfc = json.load(f)

    bert_scores = get_scores(numdecomp_claims_with_questions)

    print("*** claim decomp bert scores ****", len(bert_scores["precision"]),
          mean(bert_scores["precision"]),mean(bert_scores["recall"])
          ,mean(bert_scores["f1"]))
    # bert_scores = get_scores(claimdecomp_claims_with_questions)

    # print("*** claim decomp bert scores ****", len(bert_scores["precision"]),
    #       mean(bert_scores["precision"]),mean(bert_scores["recall"])
    #       ,mean(bert_scores["f1"]))
    # fluency_scores = fluency.score(numdecomp_claims_with_questions["questions"].values)
    # print("***numdecomp fluency scores***",mean(fluency_scores),stdev(fluency_scores))
    #print("***claimdecomp fluency scores***",fluency.score(claimdecomp_claims_with_questions["questions"].values))