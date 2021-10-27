"""
Breakdown of the spider DEV dataset on BRIDGE and RATSQL

Input: dev.json
Input: {MODEL}-predictions.txt


Out: spider/categories/{MODEL}/{category}_gold.sql
Out: spider/categories/{MODEL}/{category}_predictions.txt
"""
import json
from collections import defaultdict
from src.eval.spider.evaluate import evaluate_from_strings


gold_to_db = {}
gold_to_toks = {}

gold_to_pred = {}
pred_to_gold = {}

gold_lines = []
pred_lines = []


reserved_tokens = ['select', 'min', 'max', 'sum', 'count', 'avg', 'distinct', 'group', 'by', 'order', 'having', 'where',
                   '<', '>', '=', '!', '!=', 'in', 'not', 'between', 'and', 'or', 'like',
                   'join', 'intersect', 'union', 'except']


def load_dev():
    with open("/Users/plq360/Projects/TabularSemanticParsing/data/spider/dev.json", "r") as injson:
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


def categorize_loose():
    results_dict = defaultdict(list)

    for gold in gold_lines:
        toks = gold_to_toks[gold]
        cleaned_toks = [t for t in toks if t in reserved_tokens]

        if cleaned_toks == ['select']:
            results_dict["select"].append(gold)

        elif 'join' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'intersect' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'union' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'except' in cleaned_toks:
            results_dict["misc"].append(gold)

        elif len(cleaned_toks) == 2:
            cat = cleaned_toks[-1]
            results_dict[cat].append(gold)

        elif len(cleaned_toks) == 3:
            middle_tok = cleaned_toks[1]
            if middle_tok != 'where':
                results_dict[middle_tok].append(gold)
            else:
                third_tok = cleaned_toks[2]
                if third_tok == "=":
                    results_dict["where"].append(gold)
                elif third_tok in ['<', ">"]:
                    results_dict["maths"].append(gold)
                elif third_tok in ['!', '!=']:
                    results_dict["notequals"].append(gold)
                else:
                    results_dict[third_tok].append(gold)

        elif 'where' in cleaned_toks:
            if 'order' in cleaned_toks and 'by' in cleaned_toks:
                results_dict['order'].append(gold)
            elif cleaned_toks.count("select") >= 2:
                if "not" in cleaned_toks:
                    results_dict['not'].append(gold)
                elif "in" in cleaned_toks:
                    results_dict['in'].append(gold)
                else:
                    results_dict['misc'].append(gold)

            elif "between" in cleaned_toks:
                results_dict['between'].append(gold)

            elif "and" in cleaned_toks:
                results_dict['and'].append(gold)
            elif "or" in cleaned_toks:
                results_dict['or'].append(gold)
            elif "!" in cleaned_toks:
                results_dict['notequals'].append(gold)
            elif "<" in cleaned_toks:
                results_dict['maths'].append(gold)
            elif ">" in cleaned_toks:
                results_dict['maths'].append(gold)
            elif "=" in cleaned_toks:
                results_dict['where'].append(gold)
            else:
                results_dict['misc'].append(gold)

        elif 'having' in cleaned_toks:
            results_dict['having'].append(gold)
        elif 'group' in cleaned_toks:
            results_dict['group'].append(gold)
        elif 'order' in cleaned_toks:
            results_dict['order'].append(gold)
        else:
            second_tok = cleaned_toks[1]
            results_dict[second_tok].append(gold)

    return results_dict


def is_part_of(listA, listB):
    """Return True if listA matches or is PART OF listB"""
    res = ', '.join(map(str, listA)) in ', '.join(map(str, listB))
    if res:
        return True
    else:
        return False


def categorize_almost_strict():
    results_dict = defaultdict(list)

    where = ['select', 'where', '=']
    orderby = ['select', 'order', 'by']

    groupby1 = ['select', 'group', 'by']
    groupby2 = ['select', 'min', 'group', 'by']
    groupby3 = ['select', 'max', 'group', 'by']

    #having0 = ['select', 'group', 'by', 'having']
    having1 = ['select', 'group', 'by', 'having', 'min']
    having2 = ['select', 'group', 'by', 'having', 'max']

    maths1 = ['select', 'where', '<']
    maths2 = ['select', 'where', '>']
    maths3 = ['select', 'where', '<=']
    maths4 = ['select', 'where', '>=']

    select_in = ['select', 'where', 'in', 'select']

    like = ['select', 'where', 'like']

    select_between = ['select', 'between', 'and']

    not_equals1 = ['select', 'where', '!', '=']
    not_equals2 = ['select', 'where', '!=']

    not_in = ['select', 'where', 'not', 'in', 'select']
    not_like = ['select', 'where', 'not', 'like']

    select_and = ['select', 'where', '=', 'and', '=']
    select_or = ['select', 'where', '=', 'or', '=']


    for gold in gold_lines:
        toks = gold_to_toks[gold]
        cleaned_toks = [t for t in toks if t in reserved_tokens]

        if cleaned_toks == ['select']:
            results_dict["select"].append(gold)

        #automatic throwaway!!!!
        elif 'join' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'intersect' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'union' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'except' in cleaned_toks:
            results_dict["misc"].append(gold)

        elif len(cleaned_toks) == 2:
            cat = cleaned_toks[-1]
            results_dict[cat].append(gold)

        elif is_part_of(where, cleaned_toks):
            results_dict["where"].append(gold)

        elif is_part_of(orderby, cleaned_toks):
            results_dict["orderby"].append(gold)

        elif is_part_of(groupby1, cleaned_toks):
            results_dict["groupby"].append(gold)

        elif is_part_of(groupby2, cleaned_toks):
            results_dict["groupby"].append(gold)

        elif is_part_of(groupby3, cleaned_toks):
            results_dict["groupby"].append(gold)


        elif is_part_of(having1, cleaned_toks):
            results_dict["having"].append(gold)

        elif is_part_of(having2, cleaned_toks):
            results_dict["having"].append(gold)

        elif is_part_of(maths1, cleaned_toks):
            results_dict["maths"].append(gold)

        elif is_part_of(maths2, cleaned_toks):
            results_dict["maths"].append(gold)

        elif is_part_of(maths3, cleaned_toks):
            results_dict["maths"].append(gold)

        elif is_part_of(maths4, cleaned_toks):
            results_dict["maths"].append(gold)

        elif is_part_of(select_in, cleaned_toks):
            results_dict["in"].append(gold)

        elif is_part_of(like, cleaned_toks):
            results_dict["like"].append(gold)

        elif is_part_of(select_between, cleaned_toks):
            results_dict["between"].append(gold)

        elif is_part_of(not_equals1, cleaned_toks):
            results_dict["notequals"].append(gold)

        elif is_part_of(not_equals2, cleaned_toks):
            results_dict["notequals"].append(gold)

        elif is_part_of(not_in, cleaned_toks):
            results_dict["not"].append(gold)

        elif is_part_of(not_like, cleaned_toks):
            results_dict["notlike"].append(gold)

        elif is_part_of(select_and, cleaned_toks):
            results_dict["and"].append(gold)

        elif is_part_of(select_or, cleaned_toks):
            results_dict["or"].append(gold)

        else:
            results_dict["misc"].append(gold)

    return results_dict


def categorize_strict():
    results_dict = defaultdict(list)

    where = ['select', 'where', '=']
    orderby = ['select', 'order', 'by']

    groupby1 = ['select', 'group', 'by']
    groupby2 = ['select', 'min', 'group', 'by']
    groupby3 = ['select', 'max', 'group', 'by']

    #having0 = ['select', 'group', 'by', 'having']
    having1 = ['select', 'group', 'by', 'having', 'min']
    having2 = ['select', 'group', 'by', 'having', 'max']

    maths1 = ['select', 'where', '<']
    maths2 = ['select', 'where', '>']
    maths3 = ['select', 'where', '<=']
    maths4 = ['select', 'where', '>=']

    select_in = ['select', 'where', 'in', 'select']

    like = ['select', 'where', 'like']

    select_between = ['select', 'between', 'and']

    not_equals1 = ['select', 'where', '!', '=']
    not_equals2 = ['select', 'where', '!=']

    not_in = ['select', 'where', 'not', 'in', 'select']
    not_like = ['select', 'where', 'not', 'like']

    select_and = ['select', 'where', '=', 'and', '=']
    select_or = ['select', 'where', '=', 'or', '=']


    for gold in gold_lines:
        toks = gold_to_toks[gold]
        cleaned_toks = [t for t in toks if t in reserved_tokens]

        if cleaned_toks == ['select']:
            results_dict["select"].append(gold)

        #automatic throwaway!!!!
        elif 'join' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'intersect' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'union' in cleaned_toks:
            results_dict["misc"].append(gold)
        elif 'except' in cleaned_toks:
            results_dict["misc"].append(gold)

        elif len(cleaned_toks) == 2:
            cat = cleaned_toks[-1]
            results_dict[cat].append(gold)

        elif where == cleaned_toks:
            results_dict["where"].append(gold)

        elif orderby == cleaned_toks:
            results_dict["orderby"].append(gold)

        elif groupby1 == cleaned_toks:
            results_dict["groupby"].append(gold)

        elif groupby2 == cleaned_toks:
            results_dict["groupby"].append(gold)

        elif groupby3 == cleaned_toks:
            results_dict["groupby"].append(gold)


        elif having1 == cleaned_toks:
            results_dict["having"].append(gold)

        elif having2 == cleaned_toks:
            results_dict["having"].append(gold)

        elif maths1 == cleaned_toks:
            results_dict["maths"].append(gold)

        elif maths2 == cleaned_toks:
            results_dict["maths"].append(gold)

        elif maths3 == cleaned_toks:
            results_dict["maths"].append(gold)

        elif maths4 == cleaned_toks:
            results_dict["maths"].append(gold)

        elif select_in == cleaned_toks:
            results_dict["in"].append(gold)

        elif like == cleaned_toks:
            results_dict["like"].append(gold)

        elif select_between == cleaned_toks:
            results_dict["between"].append(gold)

        elif not_equals1 == cleaned_toks:
            results_dict["notequals"].append(gold)

        elif not_equals2 == cleaned_toks:
            results_dict["notequals"].append(gold)

        elif not_in == cleaned_toks:
            results_dict["not"].append(gold)

        elif not_like == cleaned_toks:
            results_dict["notlike"].append(gold)

        elif select_and == cleaned_toks:
            results_dict["and"].append(gold)

        elif select_or == cleaned_toks:
            results_dict["or"].append(gold)

        else:
            results_dict["misc"].append(gold)

    return results_dict


def main():
    #Step 0: Check that there's no problems with relative imports
    """
    When running breakdown.py or select_analysis.py, you need to comment out L#35 from src/eval/spider/evaluate.py
    "#from src.eval.eval_tools import eval_prediction"
    This is because eval_tools.py and evaluate.py have circular imports built into them, which causes trouble.
    """

    #Step 1 read in dev
    load_dev()

    #Step 2 read in predictions
    # BRIDGE BASELINE
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/bridge_original/8vxb/predictions.16.txt")
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/bridge_original/a97i/predictions.16.txt")
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/bridge_original/vn2k/predictions.16.txt")
    #
    # RATSQL BASELINE
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/gap.SPIDER.dev.preds.txt")
    #load_prediction_file("/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/grappa.SPIDER.dev.preds.txt")



    #SELECT! not baseline
    #load_prediction_file("v4.roberta.select.preds.txt")




    #Step 3 assert lengths and make maps
    print(f"gold lines: {len(gold_lines)}")

    print(f"pred lines: {len(pred_lines)}")

    assert len(gold_lines) == len(pred_lines)
    for pair in zip(gold_lines, pred_lines):
        g = pair[0]
        p = pair[1]
        gold_to_pred[g] = p
        pred_to_gold[p] = g

    #Step 4 sort Spider dev into categories
    #results = categorize_loose()
    #results = categorize_almost_strict()
    results = categorize_strict()

    # Step 5 Evaluate
    c = 0
    for key, item_list in results.items():
        print(f"******** {key.upper()}: {len(item_list)} examples ********")
        c += len(item_list)
        print(f"EXAMPLE: {item_list[0]}")

        input_gold = []
        input_pred = []
        for g in item_list:
            db = gold_to_db[g]
            g_line = f"{g}\t{db}"
            p = gold_to_pred[g]
            p_line = f"{p}\t{db}"

            input_gold.append(g_line)
            input_pred.append(p_line)

        entries = evaluate_from_strings(input_gold, input_pred, etype="match")
        for results in entries:
            print('{}\n'.format(results['exact']))

    print(f" COUNT: {c}")
    assert len(gold_lines) == c




main()

