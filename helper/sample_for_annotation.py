import random
import json
import os


def load_grammar_file(file):
    sample_me = []

    with open(file, "r") as injson:
        inlist = json.load(injson)
        for lil_d in inlist:
            query = lil_d["query"]
            question = lil_d["question"]
            db = lil_d["db_id"]
            rule = lil_d["rule"]
            sample_me.append({"query": query, "question": question, "db": db, "rule": rule})

    return sample_me




def main():
    the_chosen_samples = []
    the_leftover_samples = []

    data_dir = "/Users/plq360/Projects/TabularSemanticParsing/data/scfg"
    files = os.listdir(data_dir)

    for f in files:
        print(f" * {f}")
        full_path = os.path.join(data_dir, f)
        sample_me = load_grammar_file(full_path)
        samples = random.choices(sample_me, k=3)
        the_chosen_samples.append(samples[0])
        the_chosen_samples.append(samples[1])
        the_leftover_samples.append(samples[1])

    print(len(the_chosen_samples))

    sub_samples = random.choices(the_leftover_samples, k=8)
    [the_chosen_samples.append(s) for s in sub_samples]

    print(len(the_chosen_samples))

    out_path = "/Users/plq360/Projects/TabularSemanticParsing"
    new_file = os.path.join(out_path, "annotate_me_scfg.json")
    with open(new_file, 'w') as outfile:
        json.dump(the_chosen_samples, outfile, indent=4)
    print(f"**** OUT: {new_file}")




main()