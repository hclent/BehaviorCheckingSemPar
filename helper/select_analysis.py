import json
from collections import defaultdict
from src.eval.spider.evaluate import evaluate_from_strings


gold_to_db = {}
gold_to_toks = {}

gold_to_pred = {}
pred_to_gold = {}

gold_lines = []
pred_lines = []


ignore_tokens = ['select', 'from']


def load_dev():
    with open("/Users/plq360/Projects/TabularSemanticParsing/data/spider/dev.json", "r") as injson:
    #with open("/Users/plq360/Projects/TabularSemanticParsing/data/scfglong/scfg_data_select.json", "r") as injson: #V4
    #with open("/Users/plq360/Projects/TabularSemanticParsing/data/scfg.v2/scfg_data_select.json", "r") as injson: #V2
        inlist = json.load(injson)
        for lil_d in inlist:
            gold = lil_d["query"]
            db = lil_d["db_id"]
            toks = lil_d["query_toks_no_value"]
            gold_to_db[gold] = db
            gold_to_toks[gold] = toks
            gold_lines.append(gold)


def load_prediction_file(path_to_pred):
    with open(path_to_pred, "r") as infile:
        all_preds = infile.readlines()
        [pred_lines.append(l.strip("\n")) for l in all_preds]

def get_num_columns(gold_lines):
    results = defaultdict(list)

    for gold in gold_lines:
        toks = gold_to_toks[gold]
        from_indx = toks.index("from")
        table_indx = from_indx + 1
        no_table_toks = [toks[i] for i in range(0, len(toks)) if i != table_indx]
        columns = [t for t in no_table_toks if t not in ignore_tokens]

        count_commas = no_table_toks.count(",")+1

        #results[len(columns)].append(gold)
        results[count_commas].append(gold)

    return results

def main():
    """
    When running breakdown.py or select_analysis.py, you need to comment out L#35 from src/eval/spider/evaluate.py
    "#from src.eval.eval_tools import eval_prediction"
    This is because eval_tools.py and evaluate.py have circular imports built into them, which causes trouble.
    """

    #Step 1 read in dev
    load_dev()

    #Step 2 read in predictions
    #FIXME: ONLY NEED TO CHANGE PREDICTION FILE !!!!!!!
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/bridge_scfg.v2/VN2K/predictions.scfg_data_select.VN2K.txt")
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/bridge_scfg.v2/8VXB/predictions.scfg_data_select.8VXB.txt")
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/bridge_scfg.v2/A97I/predictions.scfg_data_select.A97I.txt")

    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/roberta.select.preds.txt")
    # load_prediction_file(
    #      "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/grappa.select.preds.txt")
    # load_prediction_file(
    #     "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/gap.select.preds.txt")


    #load_prediction_file("/Users/plq360/Downloads/bridge_scfg.v4/VN2K/predictions.scfg_data_select.VN2K.txt")
    #load_prediction_file("/Users/plq360/Downloads/bridge_scfg.v4/8VXB/predictions.scfg_data_select.8VXB.txt")
    #load_prediction_file("/Users/plq360/Downloads/bridge_scfg.v4/A97I/predictions.scfg_data_select.A97I.txt")
    #load_prediction_file("v4.roberta.select.preds.txt")
    #load_prediction_file("v4.grappa.select.preds.txt")
    #load_prediction_file("v4.gap.select.preds.txt")
    load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/gap.SPIDER.dev.preds.txt")

    #Step 3 assert lengths and make maps
    assert len(gold_lines) == len(pred_lines)
    for pair in zip(gold_lines, pred_lines):
        g = pair[0]
        p = pair[1]
        gold_to_pred[g] = p
        pred_to_gold[p] = g

    results = get_num_columns(gold_lines)

    # Step 5 Evaluate
    c = 0
    for key, item_list in results.items():
        print(f"******** {key}: {len(item_list)} examples ********")
        c += len(item_list)

        input_gold = []
        input_pred = []
        for i, g in enumerate(item_list):
            db = gold_to_db[g]
            g_line = f"{g}\t{db}"
            p = gold_to_pred[g]
            p_line = f"{p}\t{db}"

            input_gold.append(g_line)
            input_pred.append(p_line)
            if i < 10:
                print(f"* {g} ------> {p}")

        entries = evaluate_from_strings(input_gold, input_pred, etype="match")
        for results in entries:
            print('{}\n'.format(results['exact']))

    print(f" COUNT: {c}")
    assert len(gold_lines) == c



main()