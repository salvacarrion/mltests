import datetime
import logging
import os
import random
from pathlib import Path

import pandas as pd

from translation.autonmt_deprecated import utils
from translation.autonmt_deprecated.autonmt import commands
from translation.autonmt_deprecated.autonmt.toolkits import fairseq_entry, custom_entry

# Global vars
random.seed(123)


def preprocess(toolkit, base_path, datasets, subword_model, vocab_size, force_overwrite, interactive):
    print(f"- Preprocess: (subword_model={subword_model}; vocab_size={vocab_size})")

    # Select toolkit setup
    toolkit_setup = {
        "fairseq": {"preprocessor_fn": fairseq_entry.fairseq_preprocess},
        "custom":  {"preprocessor_fn": custom_entry.custom_preprocess},
    }
    toolkit_setup = toolkit_setup[toolkit]

    # Preprocess datasets
    for ds in datasets:  # Training dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:  # Training languages
                src_lang, trg_lang = lang_pair.split("-")

                # Get dataset path
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Preprocess
                toolkit_setup["preprocess_fn"](ds_path, src_lang, trg_lang, subword_model, vocab_size,
                                               force_overwrite, interactive=interactive)


def train(toolkit, base_path, datasets, run_name, subword_model, vocab_size, num_gpus, force_overwrite, interactive):
    print(f"- Train & Score: (run_name={run_name}, subword_model={subword_model}; vocab_size={vocab_size})")

    # Select toolkit setup
    toolkit_setup = {
        "fairseq": {"train_fn": fairseq_entry.fairseq_train},
        "custom":  {"train_fn": custom_entry.custom_train},
    }
    toolkit_setup = toolkit_setup[toolkit]

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

                # Train
                toolkit_setup["train_fn"](data_path=ds_path, run_name=run_name,
                                          src_lang=src_lang, trg_lang=trg_lang, subword_model=subword_model,
                                          vocab_size=vocab_size, model_path=None, num_gpus=num_gpus,
                                          force_overwrite=force_overwrite, interactive=interactive)


def evaluate(toolkit, base_path, train_datasets, eval_datasets, run_name, subword_model, vocab_size, beams, max_length,
             metrics, force_overwrite, interactive):
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
                                   max_length=max_length, subword_model=subword_model, vocab_size=vocab_size, metrics=metrics,
                                   force_overwrite=force_overwrite, interactive=interactive)


def evaluate_model(toolkit, base_path, model_ds_path, eval_datasets, run_name, checkpoint_path, spm_model_path, beams,
                   max_length, subword_model, vocab_size, metrics, force_overwrite, interactive):
    # model_ds_path =>  Dataset path where the trained model can be found
    # eval_ds_path => Dataset path where the evaluations datasets can be found

    toolkit_setup = {
        "fairseq": {"preprocessor_fn": fairseq_entry.fairseq_preprocess_with_vocab,
                    "translate_fn": fairseq_entry.fairseq_translate,
                    "score_fn": custom_entry.custom_score,
                    "vocab_fname": "dict.{}.txt"
                    },
        "custom":  {"preprocessor_fn": custom_entry._custom_preprocess,
                    "translate_fn": custom_entry.custom_translate,
                    "score_fn": custom_entry.custom_score,
                    "vocab_fname": "spm_vocab.{}",
                    },
    }
    toolkit_setup = toolkit_setup[toolkit]

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

                # Get model's vocabulary
                vocab_path = os.path.join(model_ds_path, "models", toolkit, "data-bin", subword_model, str(vocab_size))
                src_vocab_path = os.path.join(vocab_path, toolkit_setup["vocab_fname"].format(src_lang))
                trg_vocab_path = os.path.join(vocab_path, toolkit_setup["vocab_fname"].format(trg_lang))

                # Preprocess data (extern => encoded files were copied here)
                eval_path = os.path.join(model_ds_path, "models", toolkit, "runs", run_name, "eval", eval_name)
                eval_data_path = os.path.join(eval_path, "data")
                eval_data_bin_path = os.path.join(eval_path, "data-bin")

                # Preprocess raw evaluation files
                if not force_overwrite and os.path.exists(eval_data_bin_path):
                    print("=> Skipping preprocessing as the directory already exists")
                else:
                    toolkit_setup["preprocessor_fn"](destdir=eval_data_bin_path, src_lang=src_lang, trg_lang=trg_lang,
                                                     src_vocab_path=src_vocab_path, trg_vocab_path=trg_vocab_path,
                                                     train_fname=None, val_fname=None,
                                                     test_fname=os.path.join(eval_data_path, "test"),
                                                     src_spm_model_path=spm_model_path,
                                                     trg_spm_model_path=spm_model_path)
                for beam_width in beams:
                    # Create output path (if needed)
                    output_path = os.path.join(eval_path, "beams", f"beam_{beam_width}")
                    path = Path(output_path)
                    path.mkdir(parents=True, exist_ok=True)

                    # Translate
                    toolkit_setup["translate_fn"](data_path=eval_data_bin_path, checkpoint_path=checkpoint_path,
                                                  output_path=output_path, src_lang=src_lang, trg_lang=trg_lang,
                                                  beam_width=beam_width, max_length=max_length,
                                                  force_overwrite=force_overwrite, interactive=interactive)

                    # Score
                    toolkit_setup["score_fn"](data_path=eval_data_bin_path, output_path=eval_data_bin_path,
                                              src_lang=src_lang, trg_lang=trg_lang,
                                              trg_spm_model_path=spm_model_path, metrics=metrics,
                                              force_overwrite=force_overwrite, interactive=interactive)


def create_report(output_path, metrics, force_overwrite):
    # Create logs path
    metrics_path = os.path.join(output_path, "metrics")
    plots_path = os.path.join(output_path, "plots")
    for p in [metrics_path, plots_path]:
        path = Path(p)
        path.mkdir(parents=True, exist_ok=True)

    # Save scores
    df_metrics = save_metrics(output_path=metrics_path, metrics=metrics)

    # Plot metrics
    plot_metrics(output_path=plots_path, df_metrics=df_metrics, force_overwrite=force_overwrite)


def parse_metrics(toolkit, base_path, train_datasets, eval_datasets, run_name, subword_model, vocab_size, beams):
    print(f"- Parsing metrics....")

    # Walk through trained models
    scores = []
    for train_ds in train_datasets:  # Training dataset
        train_ds_name = train_ds["name"]
        for train_ds_size_name, train_ds_max_lines in train_ds["sizes"]:  # Training lengths
            for train_lang_pair in train_ds["languages"]:  # Training languages
                train_src_lang, train_trg_lang = train_lang_pair.split("-")

                # Train dataset path (Model path)
                train_ds_path = os.path.join(base_path, train_ds_name, train_ds_size_name, train_lang_pair)

                # Walk through Evaluations
                print(f"\tParsing model evaluations: (run_name= {run_name}, beams={str(beams)}])")
                for eval_ds in eval_datasets:  # Dataset name (to evaluate)
                    eval_ds_name = eval_ds["name"]
                    for eval_ds_size_name, eval_ds_max_lines in eval_ds["sizes"]:  # Dataset lengths (to evaluate)
                        for eval_lang_pair in eval_ds["languages"]:  # Dataset languages (to evaluate)
                            eval_src_lang, eval_trg_lang = eval_lang_pair.split("-")

                            # Get eval name
                            eval_name = "_".join([eval_ds_name, eval_ds_size_name, eval_lang_pair])

                            # Evaluation path
                            eval_path = os.path.join(train_ds_path, "models", toolkit, "runs", run_name, "eval",
                                                     eval_name)

                            # Walk through beams
                            run_scores = {
                                "train_dataset": train_ds_name, "train_size": train_ds_size_name, "train_lang_pair": train_lang_pair,
                                "eval_dataset": eval_ds_name, "eval_size": eval_ds_size_name, "eval_lang_pair": eval_lang_pair,
                                "run_name": run_name, "subword_model": subword_model, "vocab_size": vocab_size,
                                "beams": {}
                            }
                            for beam_width in beams:
                                scores_path = os.path.join(eval_path, "beams", f"beam_{beam_width}", "scores")

                                # Walk through metric files
                                beam_scores = {}
                                for m_tool, (m_fname, m_parser) in commands.METRIC_PARSERS.items():

                                    # Read file
                                    filename = os.path.join(scores_path, m_fname)
                                    if os.path.exists(filename):
                                        with open(filename, 'r') as f:
                                            m_scores = m_parser(text=f.readlines())
                                            for key, value in m_scores.items():
                                                m_name = f"{m_tool}_{key}".lower().strip()
                                                beam_scores[m_name] = value
                                    else:
                                        logging.info(f"There are no metrics for '{m_tool}'")

                                # Add beam scores
                                run_scores["beams"].update({f"beam_{str(beam_width)}": beam_scores})

                            # Add run scores
                            scores.append(run_scores)
    return scores


def save_metrics(output_path, metrics):
    # Save json metrics
    json_metrics_path = os.path.join(output_path, "metrics.json")
    utils.save_json(metrics, json_metrics_path)

    # Convert to pandas
    rows = []
    for i in range(len(metrics)):
        run_scores = dict(metrics[i])  # Copy
        beams_unrolled = {f"{beam_width}__{k}": v for beam_width in metrics[0]["beams"].keys() for k, v in
                          run_scores["beams"][beam_width].items()}
        run_scores.pop("beams")
        run_scores.update(beams_unrolled)
        rows.append(run_scores)

    # Convert to pandas
    df = pd.DataFrame(rows)
    csv_metrics_path = os.path.join(output_path, "metrics.csv")
    df.to_csv(csv_metrics_path, index=False)
    return df


def plot_metrics(output_path, df_metrics, force_overwrite):
    from translation.autonmt_deprecated.preprocess import plots

    SAVE_FIGURES = True
    SHOW_FIGURES = False

    # Set backend
    if SAVE_FIGURES:
        plots.set_non_gui_backend()
        if SHOW_FIGURES:
            raise ValueError("'save_fig' is incompatible with 'show_fig'")

    print(f"- Plotting datasets...")
    print(f"- [WARNING]: Matplotlib might miss some images if the loop is too fast")

    metric_id = "beam_1__sacrebleu_bleu"
    metric_name = metric_id.split('_')[-1].upper()
    p_fname = f"metrics__{metric_name}"
    plots.catplot(data=df_metrics, x="run_name", y=metric_id, hue="eval_dataset",
                  title=f"Model comparison", xlabel="Models", ylabel=metric_name, leyend_title=None,
                  output_dir=output_path, fname=p_fname, aspect_ratio=(8, 4), size=1.0,
                  save_fig=SAVE_FIGURES, show_fig=SHOW_FIGURES, overwrite=True, data_format="{:.2f}")


def train_and_score(base_path, train_datasets, eval_datasets, run_name_prefix, subword_models, vocab_size, force_overwrite,
                    interactive, toolkit, num_gpus, beams, max_length, metrics, output_path,
                    disable_preprocess=False, disable_train=False, disable_evaluate=False, disable_metrics=False,
                    disable_report=False):
    # Create logger
    mylogger = utils.create_logger(logs_path=os.path.join(output_path, "logs"))

    # Compute total tasks
    total_train_datasets = utils.count_datasets(eval_datasets)
    total_eval_datasets = utils.count_datasets(train_datasets)
    task_multiplier = len(subword_models)*len(vocab_size)
    total_models2train = 0 if disable_train else task_multiplier * total_train_datasets
    total_models2evaluate = 0 if disable_evaluate else task_multiplier * total_train_datasets

    mylogger.info(f"***** Train and Score *****")
    mylogger.info(f"- Total train datasets ({str(total_models2train)}):")
    mylogger.info(f"- Total evaluation datasets ({str(total_eval_datasets)}):")
    mylogger.info(f"- Task multiplier ({str(task_multiplier)}):")
    mylogger.info(f"- Total models to train ({str(total_models2train)}):")
    mylogger.info(f"- Total models to evaluate ({str(total_models2evaluate)}):")

    # Execute runs
    rows = []
    scores = []
    runs_counter = 0
    for sw_model in subword_models:
        for voc_size in vocab_size:
            # Variables
            runs_counter += 1
            run_name = f"{run_name_prefix}_{sw_model}_{voc_size}"
            row = {}

            # Summary
            mylogger.info(f"***** Starting new run *****")
            mylogger.info(f"- Summary for ({str(run_name)}):")
            mylogger.info(f"\t- Run name: {str(run_name)}")
            mylogger.info(f"\t- Run start time: {str(datetime.datetime.now())}")
            mylogger.info(f"\t- Toolkit: {str(toolkit)}")
            mylogger.info(f"\t- Metrics: {str(metrics)}")
            mylogger.info(f"\t- Num. GPUs: {str(num_gpus)}")
            mylogger.info(f"\t- Force overwrite: {str(force_overwrite)}")
            mylogger.info(f"\t- Interactive: {str(interactive)}")
            mylogger.info(f"\t- Subword model: {str(sw_model)}")
            mylogger.info(f"\t- Vocabulary size: {str(voc_size)}")
            mylogger.info(f"\t- Run number: {str(runs_counter)}")
            mylogger.info(f"\t- Training models: {str(runs_counter if total_models2train else 0)}/{str(total_models2train)}")
            mylogger.info(f"\t- Evaluating models: {str(runs_counter if total_models2evaluate else 0)}/{str(total_models2evaluate)}")

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
            if not disable_preprocess:
                kwargs = {'toolkit': toolkit, 'base_path': base_path, 'datasets': train_datasets, 'subword_model': sw_model,
                          'vocab_size': voc_size, 'force_overwrite': force_overwrite, 'interactive': interactive}
                utils.logged_task(mylogger, row, "preprocess", preprocess, **kwargs)

            # Train model
            if not disable_train:
                kwargs = {'toolkit': toolkit, 'base_path': base_path, 'datasets': train_datasets, 'run_name': run_name,
                          'subword_model': sw_model, 'vocab_size': voc_size, 'num_gpus': num_gpus,
                          'force_overwrite': force_overwrite, 'interactive': interactive}
                utils.logged_task(mylogger, row, "train", train, **kwargs)

            # Evaluate model
            if not disable_evaluate:
                kwargs = {'toolkit': toolkit, 'base_path': base_path, 'train_datasets': train_datasets,
                          'eval_datasets': eval_datasets,
                          'run_name': run_name, 'subword_model': sw_model, 'vocab_size': voc_size, 'beams': beams,
                          'max_length': max_length, 'metrics': metrics, 'force_overwrite': force_overwrite,
                          'interactive': interactive}
                utils.logged_task(mylogger, row, "evaluate", evaluate, **kwargs)

            # Get metrics
            if not disable_metrics:
                kwargs = {'toolkit': toolkit, 'base_path': base_path,
                          'train_datasets': train_datasets, 'eval_datasets': eval_datasets,
                          'run_name': run_name, 'subword_model': sw_model, 'vocab_size': voc_size, 'beams': beams}
                run_scores = utils.logged_task(mylogger, row, "parse_metrics", parse_metrics, **kwargs)
                scores += run_scores

            # Add row to rows
            row["run_end_time"] = str(datetime.datetime.now())
            rows.append(row)

            # Serve partial_runs
            try:
                # Create logs path
                path = Path(os.path.join(output_path, "logs", "runs"))
                path.mkdir(parents=True, exist_ok=True)

                # Save json
                utils.save_json(row, os.path.join(path, f"{str(runs_counter)}__{run_name}.json"))
            except Exception as e:
                mylogger.error(e)

    # Save pandas with results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_path, "logs", f"runs_summary.csv"), index=False)
    mylogger.info(f"- Total runs: {runs_counter}")

    # Create report
    if not disable_report:
        kwargs = {'output_path': output_path, 'metrics': scores, 'force_overwrite': force_overwrite}
        utils.logged_task(mylogger, {}, "create_report", create_report, **kwargs)

    mylogger.info(f"########## DONE! ##########")


if __name__ == "__main__":
    # Get base path
    if os.environ.get("LOCAL_GPU"):
        BASE_PATH = "/home/salva/Documents/datasets/nn/translation"
    else:
        BASE_PATH = "/home/scarrion/datasets/nn/translation"

    # Variables
    SUBWORD_MODELS = ["word"]
    VOCAB_SIZE = [16000]
    FORCE_OVERWRITE = True  # Overwrite whatever that already exists
    INTERACTIVE = True  # To interact with the shell if something already exists
    NUM_GPUS = 'all'  # all, 1gpu=[0]; 2gpu=[0,1];...
    TOOLKIT = "custom"  # or custom
    BEAMS = [1]
    MAX_LENGTH = 150
    METRICS = {"bleu"}
    RUN_NAME_PREFIX = "mymodel"
    OUTPUT_PATH = os.path.join(BASE_PATH, "__outputs")

    # Datasets for which to train a model
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

    # Datasets in which evaluate the different models
    EVAL_DATASETS = [
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
    ]

    # Train and Score
    train_and_score(base_path=BASE_PATH, train_datasets=TRAIN_DATASETS, eval_datasets=EVAL_DATASETS,
                    run_name_prefix=RUN_NAME_PREFIX, subword_models=SUBWORD_MODELS, vocab_size=VOCAB_SIZE,
                    force_overwrite=FORCE_OVERWRITE, interactive=INTERACTIVE,
                    toolkit=TOOLKIT, num_gpus=NUM_GPUS, beams=BEAMS, max_length=MAX_LENGTH, metrics=METRICS, output_path=OUTPUT_PATH,
                    disable_preprocess=True, disable_train=True, disable_evaluate=False, disable_metrics=False,
                    disable_report=False
                    )
