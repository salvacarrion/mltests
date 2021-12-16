import datetime
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd

from translation import utils
from translation.autonmt import commands
from translation.autonmt.toolkits import fairseq_entry

# Global vars
random.seed(123)


def preprocess(toolkit, base_path, datasets, subword_model, vocab_size, force_overwrite, interactive):
    print(f"- Preprocess: (subword_model={subword_model}; vocab_size={vocab_size})")

    # Preprocess datasets
    for ds in datasets:  # Training dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:  # Training languages
                src_lang, trg_lang = lang_pair.split("-")

                # Get dataset path
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Select toolkit
                if toolkit == "fairseq":
                    fairseq_entry.fairseq_preprocess(ds_path, src_lang, trg_lang, subword_model, vocab_size,
                                                     force_overwrite, interactive=interactive)
                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


def train(toolkit, base_path, datasets, run_name, subword_model, vocab_size, num_gpus, force_overwrite, interactive):
    print(f"- Train & Score: (run_name={run_name}, subword_model={subword_model}; vocab_size={vocab_size})")

    # Parse gpu flag
    num_gpus = None if not num_gpus or num_gpus.strip().lower() == "all" else num_gpus

    # Train models
    for ds in datasets:  # Training dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:  # Training languages
                src_lang, trg_lang = lang_pair.split("-")

                # Get dataset path
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Select toolkit
                if toolkit == "fairseq":
                    fairseq_entry.fairseq_train(data_path=ds_path, run_name=run_name, subword_model=subword_model,
                                                vocab_size=vocab_size, model_path=None, num_gpus=num_gpus,
                                                force_overwrite=force_overwrite, interactive=interactive)
                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


def evaluate(toolkit, base_path, train_datasets, eval_datasets, run_name, subword_model, vocab_size, beams, metrics,
             force_overwrite, interactive):
    print(f"- Evaluate models: (run_name={run_name}, subword_model={subword_model}; vocab_size={vocab_size})")

    # Evaluate models
    for ds in train_datasets:  # Dataset name (of the trained model)
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Dataset length (of the trained model)
            for lang_pair in ds["languages"]:  # Dataset language (of the trained model)
                src_lang, trg_lang = lang_pair.split("-")

                # Dataset path where the trained model is
                model_ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Evaluate model
                for chkpt_fname in ["checkpoint_best.pt"]:
                    checkpoint_path = os.path.join(model_ds_path, "models", toolkit, "runs", run_name, "checkpoints",
                                                   chkpt_fname)
                    spm_model_path = os.path.join(model_ds_path, "vocabs", "spm", subword_model, str(vocab_size),
                                                  f"spm_{src_lang}-{trg_lang}.model")
                    evaluate_model(toolkit=toolkit, base_path=base_path, model_ds_path=model_ds_path,
                                   eval_datasets=eval_datasets, run_name=run_name,
                                   checkpoint_path=checkpoint_path, spm_model_path=spm_model_path, beams=beams,
                                   subword_model=subword_model, vocab_size=vocab_size, metrics=metrics,
                                   force_overwrite=force_overwrite, interactive=interactive)


def evaluate_model(toolkit, base_path, model_ds_path, eval_datasets, run_name, checkpoint_path, spm_model_path, beams,
                   subword_model, vocab_size, metrics, force_overwrite, interactive):
    # model_ds_path =>  Dataset path where the trained model can be found
    # eval_ds_path => Dataset path where the evaluations datasets can be found

    print(f"- Evaluate model: (run_name= {run_name}, checkpoint_path={checkpoint_path}, beams={str(beams)}])")
    for ds in eval_datasets:  # Dataset name (to evaluate)
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Dataset lengths (to evaluate)
            for lang_pair in ds["languages"]:  # Dataset languages (to evaluate)
                src_lang, trg_lang = lang_pair.split("-")

                # Get eval name
                eval_name = "_".join([ds_name, ds_size_name, lang_pair])

                # Get extern dataset path (to evaluate)
                eval_ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Create eval paths for encoded files
                model_eval_path = os.path.join(model_ds_path, "models", toolkit, "runs", run_name, "eval", eval_name)
                model_eval_data_path = os.path.join(model_eval_path, "data")
                path = Path(model_eval_data_path)
                path.mkdir(parents=True, exist_ok=True)

                # Encode dataset using the SPM of this model
                for fname in [f"test.{src_lang}", f"test.{trg_lang}"]:
                    raw_folder = "pretokenized" if (subword_model == "word") else "splits"
                    ori_filename = os.path.join(eval_ds_path, "data", raw_folder, fname)
                    new_filename = os.path.join(model_eval_path, "data", fname)

                    # Check if the file exists
                    if not force_overwrite and os.path.exists(new_filename):
                        print("=> Skipping eval file encoding as it already exists")
                    else:
                        # Encode files
                        print(f"=> Encoding test file: {ori_filename}")
                        commands.spm_encode(spm_model_path=spm_model_path, input_file=ori_filename,
                                            output_file=new_filename)

                # Select toolkit
                if toolkit == "fairseq":
                    # Get train vocabulary
                    vocab_path = os.path.join(model_ds_path, "models", "fairseq", "data-bin", subword_model,
                                              str(vocab_size))
                    src_vocab_path = os.path.join(vocab_path, f"dict.{src_lang}.txt")
                    trg_vocab_path = os.path.join(vocab_path, f"dict.{trg_lang}.txt")

                    # Preprocess data
                    eval_path = os.path.join(model_ds_path, "models", "fairseq", "runs", run_name, "eval",
                                             eval_name)  # (extern) encoded files werw copied here
                    eval_data_path = os.path.join(eval_path, "data")
                    eval_data_bin_path = os.path.join(eval_path, "data-bin")

                    # Process raw evaluation files
                    if not force_overwrite and os.path.exists(eval_data_bin_path):
                        print("=> Skipping preprocessing as the directory already exists")
                    else:
                        fairseq_entry.fairseq_preprocess_with_vocab(data_path=eval_data_path,
                                                                    data_bin_path=eval_data_bin_path, src_lang=src_lang,
                                                                    trg_lang=trg_lang, src_vocab_path=src_vocab_path,
                                                                    trg_vocab_path=trg_vocab_path, train_fname="test",
                                                                    val_fname=None)

                    for beam_width in beams:
                        # Create output path (if needed)
                        beam_output_path = os.path.join(eval_path, "beams", f"beam_{beam_width}")
                        path = Path(beam_output_path)
                        path.mkdir(parents=True, exist_ok=True)

                        # Translate & Score
                        fairseq_entry.fairseq_translate(data_path=eval_data_bin_path, checkpoint_path=checkpoint_path,
                                                        output_path=beam_output_path, src_lang=src_lang,
                                                        trg_lang=trg_lang,
                                                        subword_model=subword_model, spm_model_path=spm_model_path,
                                                        force_overwrite=force_overwrite, beam_width=beam_width,
                                                        score=True, metrics=metrics, interactive=interactive)

                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


def logged_task(row, fn_name, fn, **kwargs):
    start_fn = time.time()
    logging.info(f"***** {fn_name.title()} started *****")
    logging.info(f"----- [{fn_name.title()}] Start time (ss.ms): {start_fn} -----")

    # Call function
    try:
        fn_status = "okay"
        fn(**kwargs)
    except Exception as e:
        logging.error(str(e))
        fn_status = str(e)

    # Get elapsed time
    end_fn = time.time()
    elapsed_fn = end_fn - start_fn
    elapsed_fn_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_fn))

    # Log time
    logging.info(f"----- [{fn_name.title()}] End time (ss.ms): {end_fn} -----")
    # logging.info(f"----- [{fn_name.title()}] Time elapsed (ss.ms): {elapsed_fn} -----")
    logging.info(f"----- [{fn_name.title()}] Time elapsed (hh:mm:ss.ms): {elapsed_fn_str} -----")
    logging.info(f"***** {fn_name.title()} ended *****")

    # Store results
    row[f"start_{fn_name}"] = start_fn
    row[f"end_{fn_name}"] = end_fn
    row[f"elapsed_{fn_name}"] = elapsed_fn
    row[f"elapsed_{fn_name}_str"] = elapsed_fn_str
    row[f"{fn_name}_status"] = fn_status


def main(base_path, train_datasets, eval_datasets, run_name, subword_models, vocab_size, force_overwrite, interactive,
         toolkit, num_gpus, beams, metrics, logs_path="logs"):
    # Setup logger
    Path(logs_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(logs_path, 'logger.log'), filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('########## LOGGER INITIATED ##########')

    # Execute runs
    rows = []
    runs_counter = 0
    for sw_model in subword_models:
        for voc_size in vocab_size:
            # Variables
            runs_counter += 1
            run_name = f"{run_name}_{sw_model}_{voc_size}"
            row = {}

            # Summary
            logging.info(f"***** Starting new run *****")
            logging.info(f"- Summary for ({str(run_name)}):")
            logging.info(f"\t- Run name: {str(run_name)}")
            logging.info(f"\t- Run start time: {str(datetime.datetime.now())}")
            logging.info(f"\t- Toolkit: {str(toolkit)}")
            logging.info(f"\t- Metrics: {str(metrics)}")
            logging.info(f"\t- Num. GPUs: {str(num_gpus)}")
            logging.info(f"\t- Force overwrite: {str(force_overwrite)}")
            logging.info(f"\t- Interactive: {str(interactive)}")
            logging.info(f"\t- Subword model: {str(sw_model)}")
            logging.info(f"\t- Vocabulary size: {str(voc_size)}")
            logging.info(f"\t- Run number: {str(runs_counter)}")

            # Add to row
            row["run_name"] = str(run_name)
            row["run_start_time"] = str(datetime.datetime.now())
            row["toolkit"] = str(toolkit)
            row["metrics"] = str(metrics)
            row["num_gpus"] = str(num_gpus)
            row["force_overwrite"] = str(force_overwrite)
            row["interactive"] = str(interactive)
            row["subword_model"] = str(sw_model)
            row["vocab_size"] = str(voc_size)
            row["run_number"] = str(runs_counter)

            # Preprocessing
            kwargs = {'toolkit': toolkit, 'base_path': base_path, 'datasets': train_datasets, 'subword_model': sw_model,
                      'vocab_size': voc_size, 'force_overwrite': force_overwrite, 'interactive': interactive}
            logged_task(row, "preprocess", preprocess, **kwargs)

            # Train model
            kwargs = {'toolkit': toolkit, 'base_path': base_path, 'datasets': train_datasets, 'run_name': run_name,
                      'subword_model': sw_model, 'vocab_size': voc_size, 'num_gpus': num_gpus,
                      'force_overwrite': force_overwrite, 'interactive': interactive}
            logged_task(row, "train", train, **kwargs)

            # Evaluate model
            kwargs = {'toolkit': toolkit, 'base_path': base_path, 'train_datasets': train_datasets,
                      'eval_datasets': eval_datasets,
                      'run_name': run_name, 'subword_model': sw_model, 'vocab_size': voc_size, 'beams': beams,
                      'metrics': metrics, 'force_overwrite': force_overwrite, 'interactive': interactive}
            logged_task(row, "evaluate", evaluate, **kwargs)

            # Add row to rows
            row["run_end_time"] = str(datetime.datetime.now())
            rows.append(row)

            # Serve partial_runs
            try:
                # Create logs path
                path = Path(logs_path, "runs")
                path.mkdir(parents=True, exist_ok=True)

                # Save json
                utils.save_json(row, os.path.join(path, f"{str(runs_counter)}__{run_name}.json"))
            except Exception as e:
                logging.error(e)

    # Save pandas with results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(logs_path, f"runs_summary.csv"), index=False)

    logging.info(f"- Total runs: {runs_counter}")
    logging.info(f"########## DONE! ##########")


if __name__ == "__main__":
    # Get base path
    if os.environ.get("LOCAL_GPU"):
        BASE_PATH = "/home/salva/Documents/datasets/nn/translation"
    else:
        BASE_PATH = "/home/scarrion/datasets/nn/translation"

    # Variables
    SUBWORD_MODELS = ["word"]  # unigram, bpe, char, or word
    VOCAB_SIZE = [16000]
    FORCE_OVERWRITE = False
    INTERACTIVE = False
    NUM_GPUS = 'all'  # all, 1gpu=[0]; 2gpu=[0,1];...
    TOOLKIT = "fairseq"
    RUN_NAME = "mymodel"
    BEAMS = [1, 5]
    METRICS = {"bleu", "chrf", "ter", "bertscore", "comet"}

    # Datasets
    TRAIN_DATASETS = [
        # {"name": "ccaligned", "sizes": [("original", None)], "languages": ["ti-en"]},
        # {"name": "commoncrawl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["cs-en", "de-en", "es-en", "fr-en"]},
        # {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "newscommentary", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
        # {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    ]

    # EVAL_DATASETS = TRAIN_DATASETS
    EVAL_DATASETS = [
        {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
    ]

    # Train and Score
    main(base_path=BASE_PATH, train_datasets=TRAIN_DATASETS, eval_datasets=EVAL_DATASETS, run_name=RUN_NAME,
         subword_models=SUBWORD_MODELS, vocab_size=VOCAB_SIZE,
         force_overwrite=FORCE_OVERWRITE, interactive=INTERACTIVE,
         toolkit=TOOLKIT, num_gpus=NUM_GPUS, beams=BEAMS, metrics=METRICS)
