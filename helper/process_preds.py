import json

output_lines = []
gold_lines = []
pred_to_orig = {}
orig_to_db = {}



with open("/Users/plq360/Projects/ratsql_robertaOUT/between.bert_run_true_1-step20100.eval.txt", "r") as infile:
    indict = json.load(infile)
    list_of_dicts = indict["per_item"]

    for lil_d in list_of_dicts:
        pred = lil_d["predicted"]
        output_lines.append(pred)
        gold = lil_d["gold"]
        pred_to_orig[pred] = gold

with open("/Users/plq360/Projects/scfg.v2/scfg_data_between.json", "r") as injson:
    inlist = json.load(injson)
    for lil_d in inlist:
        gold = lil_d["query"]
        db = lil_d["db_id"]
        orig_to_db[gold] = db
        gold_lines.append(f"{gold}\t{db}")


with open("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/gold/between_gold.sql", "w") as outgold:
    for i, l in enumerate(gold_lines):
        if i != len(gold_lines):
            outgold.write(f"{l}\n")
        else:
            outgold.write(f"{l}")


with open("predictions.between.txt", "w") as outfile:
    for i, l in enumerate(output_lines):
        gold = pred_to_orig[l]
        db = orig_to_db[gold]

        if i != len(output_lines):
            outfile.write(f"{l}\n")
        else:
            outfile.write(f"{l}")

