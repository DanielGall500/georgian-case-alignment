from grewtse.pipeline import GrewTSEPipe
import pandas as pd
import os

"""
This handles the creation of the minimal-pair syntactic tests from the Georgian Language Corpus (GLC) Treebank.
For this to work, you need to have the GLC train, dev, and test datasets downloaded, an example of which shown below.

The Grew-TSE pipeline generates the pairs based on the configuration provided.
In each case, it will find sentences matching a construction C and change a single morphosyntactic feature.
Full details and examples are shown in the paper attached to this work.

The below code shows the full process, however for a simpler and more intuitive breakdown of what Grew-TSE is doing, please refer to its documentation at https://grew-tse.readthedocs.io/
"""

# all treebank files to be used for the minimal-pair generation
TREEBANKS_KARTULI = [
    "./treebanks/ka_glc-ud-train.conllu",
    "./treebanks/ka_glc-ud-dev.conllu",
    "./treebanks/ka_glc-ud-test.conllu",
]
OUTPUT_DIR = "./output"
LEXICON_FILE = "./output/georgian-lexicon.csv"


def create_config(
    query: str, dep_node: str, convert_case_to: str, task_prefix: str
) -> dict:
    task_name = f"{task_prefix}-{dep_node}-to-{convert_case_to}"
    results_dir = f"{OUTPUT_DIR}/{task_prefix}"

    # create the folder where the results should be stored
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    return {
        "treebanks": TREEBANKS_KARTULI,
        "grew_query": query,
        "dependency_node": dep_node,
        "apply_leading_space": False,  # typically False for MLM, True for NTP
        "output_dataset": f"{results_dir}/{task_name}.csv",
        "alternative_morph_features": {"case": convert_case_to},
        "save_lexicon_to": f"{OUTPUT_DIR}/{LEXICON_FILE}",
        "task_name": task_name,
    }


def run_config(config: dict):
    grewtse = GrewTSEPipe()

    if not os.path.isfile(config["save_lexicon_to"]):
        lexicon = grewtse.parse_treebank(config["treebanks"])
        lexicon.to_csv(config["save_lexicon_to"])
    else:
        grewtse.load_lexicon(config["save_lexicon_to"], config["treebanks"])

    masked_df = grewtse.generate_masked_dataset(
        config["grew_query"], config["dependency_node"]
    )

    # Generate minimal pairs dataset
    mp_dataset = grewtse.generate_minimal_pair_dataset(
        config["alternative_morph_features"],
        # ood_pairs= config["ood_pairs"],
        has_leading_whitespace=config["apply_leading_space"],
    )
    mp_dataset.to_csv(config["output_dataset"])

    # output variables
    task = config["task_name"]
    structures_masked = masked_df.shape[0]
    mps_found = mp_dataset.shape[0]
    return task, structures_masked, mps_found


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Intransitive Constructions --
    task_prefix = "ka-intransitive"

    intransitive_query = """
            pattern {
              V [upos="VERB"];
              SUBJ [Case="Nom"];
              V -[nsubj]-> SUBJ;
            }

            without {
              V [upos="VERB"];
              V -[nsubj]-> SUBJ;
              V -[obj]-> OBJ; 
            }
        """

    config_in_to_erg = create_config(
        query=intransitive_query,
        dep_node="SUBJ",
        convert_case_to="Dat",
        task_prefix=task_prefix,
    )

    config_in_to_dat = create_config(
        query=intransitive_query,
        dep_node="SUBJ",
        convert_case_to="Erg",
        task_prefix="ka-intransitive-to-ERG",
    )

    # == TRANSITIVE (S1) ==

    # -- S1 (Nom-Dat)
    trns_s1_query = """
            pattern {
                V [upos="VERB"];
                SUBJ [Case="Nom"];
                OBJ [Case="Dat"];
                V -[nsubj]-> SUBJ;
                V -[obj]-> OBJ;
            }
            """
    task_prefix_s1 = "ka-transitive-S1"

    # -- S1 (Nom-Dat), Convert SUBJ->Ergative --
    config_trns_s1_subj_to_erg = create_config(
        query=trns_s1_query,
        dep_node="SUBJ",
        convert_case_to="Erg",
        task_prefix=task_prefix_s1,
    )

    # -- S1 (Nom-Dat), Convert SUBJ->Dative --
    config_trns_s1_subj_to_dat = create_config(
        query=trns_s1_query,
        dep_node="SUBJ",
        convert_case_to="Dat",
        task_prefix=task_prefix_s1,
    )

    # -- S1 (Nom-Dat), Convert OBJ->Nominative --
    config_trns_s1_obj_to_nom = create_config(
        query=trns_s1_query,
        dep_node="OBJ",
        convert_case_to="Nom",
        task_prefix=task_prefix_s1,
    )

    # -- S1 (Nom-Dat), Convert OBJ->Ergative --
    config_trns_s1_obj_to_erg = create_config(
        query=trns_s1_query,
        dep_node="OBJ",
        convert_case_to="Erg",
        task_prefix=task_prefix_s1,
    )

    # == TRANSITIVE (S2) ==

    # -- S2 (Erg-Nom)
    trns_s2_query = """
            pattern {
                V [upos="VERB"];
                SUBJ [Case="Erg"];
                OBJ [Case="Nom"];
                V -[nsubj]-> SUBJ;
                V -[obj]-> OBJ;
            }
            """
    task_prefix_s2 = "ka-transitive-S2"

    # -- S2 (Erg-Nom), Convert SUBJ->Nominative --
    config_trns_s2_subj_to_nom = create_config(
        query=trns_s2_query,
        dep_node="SUBJ",
        convert_case_to="Nom",
        task_prefix=task_prefix_s2,
    )

    # -- S2 (Erg-Nom), Convert SUBJ->Dative --
    config_trns_s2_subj_to_dat = create_config(
        query=trns_s2_query,
        dep_node="SUBJ",
        convert_case_to="Dat",
        task_prefix=task_prefix_s2,
    )

    # -- S2 (Erg-Nom), Convert OBJ->Ergative --
    config_trns_s2_obj_to_erg = create_config(
        query=trns_s2_query,
        dep_node="OBJ",
        convert_case_to="Erg",
        task_prefix=task_prefix_s2,
    )

    # -- S2 (Erg-Nom), Convert OBJ->Dative --
    config_trns_s2_obj_to_dat = create_config(
        query=trns_s2_query,
        dep_node="OBJ",
        convert_case_to="Dat",
        task_prefix=task_prefix_s2,
    )

    # == TRANSITIVE (S3) ==

    # -- S3 (Dat-Nom)
    trns_s3_query = """
            pattern {
                V [upos="VERB"];
                SUBJ [Case="Dat"];
                OBJ [Case="Nom"];
                V -[nsubj]-> SUBJ;
                V -[obj]-> OBJ;
            }
            """
    task_prefix_s3 = "ka-transitive-S3"

    # -- S3 (Dat-Nom), Convert SUBJ->Nominative --
    config_trns_s3_subj_to_nom = create_config(
        query=trns_s3_query,
        dep_node="SUBJ",
        convert_case_to="Nom",
        task_prefix=task_prefix_s3,
    )

    # -- S3 (Dat-Nom), Convert SUBJ->Ergative --
    config_trns_s3_subj_to_erg = create_config(
        query=trns_s3_query,
        dep_node="SUBJ",
        convert_case_to="Erg",
        task_prefix=task_prefix_s3,
    )

    # -- S3 (Dat-Nom), Convert OBJ->Dative --
    config_trns_s3_obj_to_dat = create_config(
        query=trns_s3_query,
        dep_node="OBJ",
        convert_case_to="Dat",
        task_prefix=task_prefix_s3,
    )

    # -- S3 (Dat-Nom), Convert OBJ->Ergative --
    config_trns_s3_obj_to_erg = create_config(
        query=trns_s3_query,
        dep_node="OBJ",
        convert_case_to="Erg",
        task_prefix=task_prefix_s3,
    )

    all_intransitive_configs_nom = [
        config_in_to_erg,
        config_in_to_dat,
    ]

    all_transitive_configs_nom_dat = [
        config_trns_s1_subj_to_erg,
        config_trns_s1_subj_to_dat,
        config_trns_s1_obj_to_nom,
        config_trns_s1_obj_to_erg,
    ]

    all_transitive_configs_erg_nom = [
        config_trns_s2_subj_to_nom,
        config_trns_s2_subj_to_dat,
        config_trns_s2_obj_to_erg,
        config_trns_s2_obj_to_dat,
    ]

    all_transitive_configs_dat_nom = [
        config_trns_s3_subj_to_nom,
        config_trns_s3_subj_to_erg,
        config_trns_s3_obj_to_dat,
        config_trns_s3_obj_to_erg,
    ]

    all_verbal_paradigm_configs = [
        all_intransitive_configs_nom,
        all_transitive_configs_nom_dat,
        all_transitive_configs_erg_nom,
        all_transitive_configs_dat_nom,
    ]

    results = {"task_name": [], "structures_masked": [], "minimal_pairs_found": []}

    # run this for each of the configs to carry out the
    # process of generating minimal-pair sentences
    # here the example for
    for verbal_paradigm_configs in all_verbal_paradigm_configs:
        for config in verbal_paradigm_configs:
            print("Parsing...")

            task_name, structures_masked, minimal_pairs_found = run_config(config)

            results["task_name"].append(task_name)
            results["structures_masked"].append(structures_masked)
            results["minimal_pairs_found"].append(minimal_pairs_found)
            print(f"Completed parsing {task_name}.")
            print("----")

        results = pd.DataFrame(results)
        results.to_csv(f"{OUTPUT_DIR}/meta.csv", mode="a")


if __name__ == "__main__":
    main()
