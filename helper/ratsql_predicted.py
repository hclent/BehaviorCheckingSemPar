import json



def makePredsFile(inputfile, outputfile):
    output_lines = []
    with open(inputfile, "r") as infile:
        indict = json.load(infile)
        list_of_dicts = indict["per_item"]
        for lil_d in list_of_dicts:
            pred = lil_d["predicted"]
            output_lines.append(pred)

    with open(outputfile, "w") as outfile:
        for l in output_lines:
            outfile.write(f"{l}\n")
    print(f"* {outputfile} done!")


#makePredsFile("/Users/plq360/Projects/ratsql_robertaOUT/select.bert_run_true_1-step20100.eval", "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/roberta.select.preds.txt")
#makePredsFile("/Users/plq360/Projects/ratsql_grappaOUT/select.bert_run_true_1-step39100.eval", "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/grappa.select.preds.txt")
#makePredsFile("/Users/plq360/Projects/GAP_OUT/select.bart_run_1_true_1-step41000.eval", "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/gap.select.preds.txt")


#### For spider dev analysis
#makePredsFile("/Users/plq360/Downloads/gap_original/gap_predictions_scfg_data_SPIDER.eval", "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/gap.SPIDER.dev.preds.txt")
#makePredsFile("/Users/plq360/Downloads/grappa_original/grappa_predictions_scfg_data_SPIDER.eval", "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/grappa.SPIDER.dev.preds.txt")
#makePredsFile("/Users/plq360/Downloads/roberta_original/roberta_predictions_scfg_data_SPIDER.eval", "/Users/plq360/Projects/TabularSemanticParsing/src/eval/results/ratsql/roberta.SPIDER.dev.preds.txt")


#### For select analysis
#makePredsFile("/Users/plq360/Downloads/roberta_scfg.v4/roberta_predictions_scfg_data_select.eval", "v4.roberta.select.preds.txt")
#makePredsFile("/Users/plq360/Downloads/grappa_scfg.v4/grappa_predictions_scfg_data_select.eval", "v4.grappa.select.preds.txt")
#makePredsFile("/Users/plq360/Downloads/gap_scfg.v4/gap_predictions_scfg_data_select.eval", "v4.gap.select.preds.txt")


#### For MIN
makePredsFile("/Users/plq360/Projects/ratsql_robertaOUT/min.bert_run_true_1-step20100.eval.txt", "v4.roberta.min.preds.txt")
makePredsFile("/Users/plq360/Projects/ratsql_grappaOUT/min.bert_run_true_1-step39100.eval.txt", "v4.grappa.min.preds.txt")
makePredsFile("/Users/plq360/Projects/GAP_OUT/min.bart_run_1_true_1-step41000.eval.txt", "v4.gap.min.preds.txt")

#### For MAX
makePredsFile("/Users/plq360/Projects/ratsql_robertaOUT/max.bert_run_true_1-step20100.eval.txt", "v4.roberta.max.preds.txt")
makePredsFile("/Users/plq360/Projects/ratsql_grappaOUT/max.bert_run_true_1-step39100.eval.txt", "v4.grappa.max.preds.txt")
makePredsFile("/Users/plq360/Projects/GAP_OUT/max.bart_run_1_true_1-step41000.eval.txt", "v4.gap.max.preds.txt")

#### For BETWEEN
makePredsFile("/Users/plq360/Projects/ratsql_robertaOUT/between.bert_run_true_1-step20100.eval.txt", "v4.roberta.between.preds.txt")
makePredsFile("/Users/plq360/Projects/ratsql_grappaOUT/between.bert_run_true_1-step39100.eval.txt", "v4.grappa.between.preds.txt")
makePredsFile("/Users/plq360/Projects/GAP_OUT/between.bart_run_1_true_1-step41000.eval.txt", "v4.gap.between.preds.txt")




