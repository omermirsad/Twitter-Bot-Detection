Okay, here is the complete set of Python scripts and a shell script to run the full pipeline: MLM pre-training starting from `bert-base-uncased`, followed by E5-style contrastive fine-tuning.

**1. Requirements (`requirements.txt`)**

```txt
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
sentence-transformers>=2.2.2
evaluate # For HF Trainer MLM eval metrics (optional in MLM script)
pandas # For data prep scripts
scikit-learn # For data prep scripts

# Optional
deepspeed>=0.9.5 # For large scale MLM
tensorboard>=2.13.0 # For logging
wandb>=0.15.0 # For logging
```

**2. MLM Data Preparation Script (`prepare_mlm_data.py`)**

```python
# prepare_mlm_data.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare data for MLM pretraining (single text column)")
    parser.add_argument("--input_files", type=str, nargs="+", required=True,
                        help="Input text files (one document/sentence per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save train.csv and validation.csv")
    parser.add_argument("--validation_split", type=float, default=0.01, # Smaller split often ok for MLM eval
                        help="Percentage for validation split (e.g., 0.01 for 1%)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Name for the text column in the output CSV")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for shuffling and splitting")
    args = parser.parse_args()

    all_texts = []
    logger.info("Loading text files for MLM...")
    for file_path in args.input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Filter out empty lines and strip whitespace
                texts = [line.strip() for line in f if line.strip()]
                all_texts.extend(texts)
                logger.info(f"  Loaded {len(texts)} lines from {file_path}")
        except FileNotFoundError:
            logger.warning(f"File not found {file_path}, skipping.")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}, skipping.")

    if not all_texts:
        logger.error("No text data loaded. Exiting.")
        return

    logger.info(f"Total loaded text examples: {len(all_texts)}")

    # Create a pandas DataFrame
    df = pd.DataFrame(all_texts, columns=[args.text_column])
    logger.info(f"Created DataFrame with shape: {df.shape}")

    # Split dataset
    logger.info(f"Splitting data: {(1-args.validation_split)*100:.1f}% train, {args.validation_split*100:.1f}% validation...")
    if len(df) < 2:
        logger.error("Not enough data to split into training and validation sets. Need at least 2 examples.")
        return

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=args.validation_split,
            random_state=args.random_seed,
            shuffle=True
        )
    except ValueError as e:
        logger.error(f"Error during train/test split: {e}. Check validation_split value and data size.")
        # Fallback: Use all data for training if split fails and validation is not strictly required downstream
        logger.warning("Using all data for training as split failed.")
        train_df = df
        val_df = pd.DataFrame(columns=[args.text_column]) # Empty validation set


    # Save to disk
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, "train.csv")
        val_path = os.path.join(args.output_dir, "validation.csv")

        train_df.to_csv(train_path, index=False, encoding='utf-8')
        if not val_df.empty:
            val_df.to_csv(val_path, index=False, encoding='utf-8')
        else:
            # Create an empty validation file if needed by downstream processes
             with open(val_path, 'w', encoding='utf-8') as f:
                 f.write(f"{args.text_column}\n")


        logger.info(f"Saved {len(train_df)} training examples to {train_path}")
        logger.info(f"Saved {len(val_df)} validation examples to {val_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    main()
```

**3. MLM Pre-training Script (`run_mlm_pretraining.py`)**

```python
# run_mlm_pretraining.py

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import datasets
import torch
import transformers
from datasets import load_dataset, DatasetDict
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Version checks
check_min_version("4.30.0")
require_version("datasets>=2.12.0", "To fix: pip install -r requirements.txt")

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models (e.g., bert-base-uncased)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface/token`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512, # Default to 512 for E5-style models
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the full texts (for json/csv files)."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False, # Process whole dataset for better packing
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


def main():
    # --- 1. Parse arguments ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 2. Setup logging levels for libraries ---
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Log the configuration
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # --- 3. Detect Checkpoints ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # Check if output dir contains model files, not just logs etc.
            model_files_present = any(fname.endswith((".bin", ".safetensors")) for fname in os.listdir(training_args.output_dir))
            if model_files_present:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            else:
                 logger.info(f"Output directory ({training_args.output_dir}) exists but contains no checkpoints or model files. Proceeding.")

        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # --- 4. Set Seed ---
    set_seed(training_args.seed)

    # --- 5. Load Data ---
    raw_datasets = DatasetDict()
    if data_args.dataset_name is not None:
        logger.info(f"Loading dataset from Hub: {data_args.dataset_name}")
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            # streaming=True, # Consider streaming for very large datasets
        )
        # Handle potential missing validation split in Hub dataset
        if "validation" not in raw_datasets.keys():
            logger.info(f"Validation split not found in Hub dataset. Splitting train set: {100-data_args.validation_split_percentage}% train, {data_args.validation_split_percentage}% validation.")
            if "train" not in raw_datasets:
                 raise ValueError("Hub dataset must contain a 'train' split if 'validation' is missing.")
            # Split the training dataset
            split_datasets = raw_datasets["train"].train_test_split(
                test_size=data_args.validation_split_percentage / 100.0,
                seed=training_args.seed,
                load_from_cache_file=not data_args.overwrite_cache
            )
            raw_datasets["train"] = split_datasets["train"]
            raw_datasets["validation"] = split_datasets["test"]

    elif data_args.train_file is not None:
        logger.info(f"Loading data from files: train={data_args.train_file}, validation={data_args.validation_file}")
        data_files = {}
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
        if extension not in ["csv", "json", "txt", "jsonl"]:
             raise ValueError(f"Unsupported train file extension: {extension}")

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            val_extension = data_args.validation_file.split(".")[-1]
            if val_extension not in ["csv", "json", "txt", "jsonl"]:
                 raise ValueError(f"Unsupported validation file extension: {val_extension}")
            if val_extension != extension and not (extension in ["json", "jsonl"] and val_extension in ["json", "jsonl"]):
                 logger.warning(f"Train ({extension}) and validation ({val_extension}) file extensions differ.")
        else:
            logger.info("No validation file provided.")


        # Adjust loading format based on extension
        load_format = extension
        if extension == "txt":
            load_format = "text"
        elif extension == "jsonl":
             load_format = "json" # datasets library uses 'json' for jsonl

        raw_datasets = load_dataset(
            load_format,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        # If no validation data is specified, split the training data.
        if "validation" not in raw_datasets.keys():
            if "train" not in raw_datasets:
                 raise ValueError("No 'train' split found to create validation split from.")
            logger.info(f"Validation file not provided. Splitting train set: {100-data_args.validation_split_percentage}% train, {data_args.validation_split_percentage}% validation.")
            # Shuffle before splitting is implicitly handled by train_test_split
            split_datasets = raw_datasets["train"].train_test_split(
                test_size=data_args.validation_split_percentage / 100.0,
                seed=training_args.seed,
                load_from_cache_file=not data_args.overwrite_cache
            )
            raw_datasets["train"] = split_datasets["train"]
            raw_datasets["validation"] = split_datasets["test"]
    else:
        # Neither dataset_name nor train_file was provided
        raise ValueError("You must specify either `--dataset_name` or `--train_file`.")

    logger.info(f"Raw datasets loaded: {raw_datasets}")

    # --- 6. Load Model & Tokenizer ---
    logger.info("Loading pretrained model and tokenizer...")
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        logger.info(f"Loaded config '{model_args.config_name}'")
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        logger.info(f"Loaded config for model '{model_args.model_name_or_path}'")
    else:
        # This case should ideally not be hit if model_name_or_path is required, but for safety:
        raise ValueError("Must provide model_name_or_path or config_name")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        logger.info(f"Loaded tokenizer '{model_args.tokenizer_name}'")
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        logger.info(f"Loaded tokenizer for model '{model_args.model_name_or_path}'")
    else:
        raise ValueError("Must provide model_name_or_path or tokenizer_name")

    logger.info(f"Loading model '{model_args.model_name_or_path}' for Masked LM pre-training.")
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    logger.info(f"Model loaded: {type(model)}")

    # Resize token embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(f"Resizing token embeddings from {embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # --- 7. Preprocess Data ---
    logger.info("Preprocessing datasets...")
    # Determine text column name
    if training_args.do_train and "train" in raw_datasets:
        column_names = list(raw_datasets["train"].features)
    elif training_args.do_eval and "validation" in raw_datasets:
        column_names = list(raw_datasets["validation"].features)
    elif "train" in raw_datasets: # Fallback if not doing train/eval but need columns
         column_names = list(raw_datasets["train"].features)
    elif "validation" in raw_datasets:
         column_names = list(raw_datasets["validation"].features)
    else:
         raise ValueError("Cannot determine dataset columns. No 'train' or 'validation' split found.")

    if data_args.text_column_name is None or data_args.text_column_name not in column_names:
         logger.warning(f"--text_column_name '{data_args.text_column_name}' not found in dataset columns: {column_names}. Trying to use 'text'.")
         if 'text' in column_names:
             text_column_name = 'text'
         else:
             # If 'text' isn't found either, raise error.
             raise ValueError(f"Cannot find text column. Specify --text_column_name among {column_names}")
    else:
        text_column_name = data_args.text_column_name
    logger.info(f"Using '{text_column_name}' as the text column.")

    # Set max_seq_length
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024: # Safety hard cap
            logger.warning(f"Tokenizer max length is {max_seq_length}. Setting max_seq_length to 1024.")
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    logger.info(f"Setting max sequence length to {max_seq_length}")


    # Tokenization function
    def tokenize_function(examples: Dict[str, List]) -> Dict[str, List]:
        # Ensure the text column exists and contains strings
        if text_column_name not in examples:
             raise ValueError(f"Text column '{text_column_name}' not found in batch keys: {list(examples.keys())}")
        texts = examples[text_column_name]
        # Handle potential None values or non-string types gracefully
        processed_texts = [str(t) if t is not None else "" for t in texts]
        return tokenizer(
            processed_texts,
            return_special_tokens_mask=True # Needed for MLM collator
        )

    # Grouping function (for line_by_line=False)
    def group_texts(examples: Dict[str, List]) -> Dict[str, List]:
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder, adjust as needed (add padding?)
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        else:
             # Handle cases where a batch results in less than max_seq_length total tokens
             # Option 1: Keep it (will be padded by collator if needed) - chosen here
             # Option 2: Discard it (potentially losing data)
             pass


        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        # Ensure the result is not empty if total_length was < max_seq_length but > 0
        if total_length > 0 and not result.get("input_ids"):
             result = {k: [t[0:total_length]] for k, t in concatenated_examples.items()}


        # Add attention_mask back if it was removed during concatenation (it shouldn't be if tokenizer returns it)
        # This step might be redundant if tokenizer correctly returns attention_mask
        if "attention_mask" not in result and "input_ids" in result:
             result["attention_mask"] = [[1] * len(ids) for ids in result["input_ids"]]

        # Ensure 'labels' field is present for MLM training if not already created by tokenizer/grouping
        # The DataCollatorForLanguageModeling will actually create the labels by masking input_ids
        # result["labels"] = result["input_ids"].copy() # Not needed here, collator handles it

        return result

    # Execute preprocessing
    with training_args.main_process_first(desc="dataset map tokenization"):
        if data_args.line_by_line:
             logger.info("Tokenizing dataset line by line...")
             # Tokenize each line separately
             padding = "max_length" if data_args.pad_to_max_length else False
             def tokenize_line_by_line(examples: Dict[str, List]) -> Dict[str, List]:
                 lines = [str(line).strip() for line in examples[text_column_name] if str(line).strip()]
                 if not lines:
                     # Return empty dict structure if batch has no valid lines
                     # Need to know the expected output keys from the tokenizer
                     # Assuming 'input_ids', 'attention_mask', 'special_tokens_mask'
                     return {"input_ids": [], "attention_mask": [], "special_tokens_mask": []}
                 return tokenizer(
                     lines,
                     padding=padding,
                     truncation=True,
                     max_length=max_seq_length,
                     return_special_tokens_mask=True,
                 )
             tokenized_datasets = raw_datasets.map(
                 tokenize_line_by_line,
                 batched=True,
                 num_proc=data_args.preprocessing_num_workers,
                 remove_columns=[text_column_name], # Keep other columns if they exist? No, remove all original.
                 # remove_columns=list(raw_datasets["train"].features) if "train" in raw_datasets else None, # Safer removal
                 load_from_cache_file=not data_args.overwrite_cache,
                 desc="Running tokenizer on dataset line_by_line",
             )
             # Ensure consistent columns after map if remove_columns failed
             if "train" in raw_datasets and text_column_name in tokenized_datasets["train"].features:
                 tokenized_datasets = tokenized_datasets.remove_columns(text_column_name)

        else:
            logger.info("Tokenizing and grouping texts...")
            # Tokenize first
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, # Remove all original columns
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            logger.info(f"Tokenized dataset features: {tokenized_datasets}")

            # Then group texts
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True, # Process batches of examples for grouping
                batch_size=1000, # Default batch size for map
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )
            logger.info(f"Grouped dataset features: {tokenized_datasets}")

    logger.info(f"Processed datasets: {tokenized_datasets}")

    # --- 8. Prepare Datasets for Trainer ---
    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            logger.info(f"Limiting training set to {max_train_samples} samples.")
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Final Train dataset size: {len(train_dataset)}")

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            logger.info(f"Limiting evaluation set to {max_eval_samples} samples.")
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Final Eval dataset size: {len(eval_dataset)}")


    # --- 9. Data Collator ---
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None
    )
    logger.info(f"Using DataCollatorForLanguageModeling with mlm_probability={data_args.mlm_probability}, pad_to_multiple_of={8 if pad_to_multiple_of_8 else None}")

    # --- 10. Initialize Trainer ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics and preprocess_logits_for_metrics can be added here if needed
        # for more detailed MLM evaluation beyond loss/perplexity.
    )
    logger.info("Trainer initialized.")

    # --- 11. Training ---
    if training_args.do_train:
        logger.info("*** Starting MLM Pre-training ***")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_steps if training_args.max_steps > 0 else 'Calculated by Trainer'}") # Trainer calculates if max_steps <= 0

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            logger.info(f"Resuming from explicit checkpoint: {checkpoint}")
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            logger.info(f"Resuming from detected checkpoint: {checkpoint}")

        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            # Save the final HF model
            logger.info("Saving final Hugging Face model...")
            trainer.save_model() # Saves tokenizer too

            metrics = train_result.metrics
            if train_dataset:
                 max_train_samples = (
                     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                 )
                 metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            logger.info(f"MLM Pre-training finished. Hugging Face model saved to {training_args.output_dir}")

        except Exception as e:
             logger.error(f"Error during training: {e}", exc_info=True)
             raise e

    # --- 12. Evaluation ---
    if training_args.do_eval:
        logger.info("*** Evaluate MLM Performance ***")
        if eval_dataset is None:
            logger.warning("Evaluation dataset is not available. Skipping evaluation.")
        else:
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Batch size = {training_args.eval_batch_size}")
            try:
                metrics = trainer.evaluate()

                if eval_dataset:
                    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                try:
                    perplexity = math.exp(metrics["eval_loss"])
                except OverflowError:
                    perplexity = float("inf")
                metrics["perplexity"] = perplexity
                logger.info(f"  Perplexity: {perplexity}")

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
            except Exception as e:
                 logger.error(f"Error during evaluation: {e}", exc_info=True)
                 # Decide if this should halt the script or just log the error
                 # raise e # Uncomment to halt on evaluation error


    # --- 13. Convert to Sentence Transformer format ---
    # Determine the path of the model to convert (best checkpoint or final output)
    final_model_path = training_args.output_dir
    best_model_checkpoint_path = trainer.state.best_model_checkpoint
    if training_args.load_best_model_at_end and best_model_checkpoint_path:
        final_model_path = best_model_checkpoint_path
        logger.info(f"Using best model checkpoint from {final_model_path} for Sentence Transformer conversion.")
    elif not training_args.do_train and os.path.isdir(model_args.model_name_or_path):
         # If only evaluating a local model, use that path
         final_model_path = model_args.model_name_or_path
         logger.info(f"No training performed, using input model path {final_model_path} for potential conversion.")
    else:
         logger.info(f"Using final model output path {final_model_path} for Sentence Transformer conversion.")


    # Check if the target model path actually contains model files before attempting conversion
    model_files_exist = False
    if os.path.isdir(final_model_path):
        model_files_exist = any(fname.endswith((".bin", ".safetensors", "pytorch_model.bin")) for fname in os.listdir(final_model_path))

    if model_files_exist:
        logger.info(f"Attempting to convert model at '{final_model_path}' to Sentence Transformers format...")
        try:
            from sentence_transformers import SentenceTransformer, models

            # Step 1: Load the Hugging Face transformer model from the saved path
            logger.info(f"Loading base transformer model from: {final_model_path}")
            word_embedding_model = models.Transformer(final_model_path)
            logger.info("Base transformer model loaded successfully.")

            # Step 2: Add a pooling layer (Mean Pooling is common, CLS is also used e.g., in E5 fine-tuning)
            pooling_mode_mean = True # Default to mean pooling
            pooling_mode_cls = False
            logger.info(f"Adding Pooling layer (mean={pooling_mode_mean}, cls={pooling_mode_cls})")
            pooling_model = models.Pooling(
                word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=pooling_mode_mean,
                pooling_mode_cls_token=pooling_mode_cls,
                pooling_mode_max_tokens=False, # Max pooling usually not used for sentence embeddings
            )
            logger.info("Pooling layer added.")

            # Step 3: Create the Sentence Transformer model
            st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            logger.info("Sentence Transformer model object created.")

            # Step 4: Save the Sentence Transformer model to a subdirectory
            st_output_path = os.path.join(training_args.output_dir, "sentence-transformer-format")
            os.makedirs(st_output_path, exist_ok=True)
            logger.info(f"Saving Sentence Transformer model to: {st_output_path}")
            st_model.save(st_output_path)
            logger.info(f"Model saved successfully in Sentence Transformers format at: {st_output_path}")
            logger.info("This model can now be used as input for contrastive fine-tuning or directly for embeddings.")

        except ImportError:
            logger.warning("`sentence-transformers` library not found. Skipping conversion.")
            logger.warning("Install it with `pip install sentence-transformers` to enable this feature.")
        except Exception as e:
            logger.error(f"Error during Sentence Transformers conversion: {e}", exc_info=True)
    else:
         logger.warning(f"No model files found at determined path '{final_model_path}'. Skipping Sentence Transformer conversion.")


    # --- 14. Push to Hub / Model Card ---
    # Optional: Add logic for push_to_hub and create_model_card if needed,
    # using trainer.push_to_hub() and trainer.create_model_card()
    logger.info("MLM Pre-training script finished.")


if __name__ == "__main__":
    main()
```

**4. Contrastive Data Preparation Script (`prepare_contrastive_data.py`)**

```python
# prepare_contrastive_data.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Contrastive fine-tuning (query, positive columns)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file (e.g., TSV/CSV with query<delimiter>positive_passage)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save train.csv and validation.csv")
    parser.add_argument("--delimiter", type=str, default="\t", help="Delimiter for input file")
    parser.add_argument("--query_column_out", type=str, default="query", help="Output column name for queries")
    parser.add_argument("--positive_column_out", type=str, default="positive", help="Output column name for positives")
    parser.add_argument("--validation_split", type=float, default=0.01, # Can be small if you have a dedicated large eval set
                        help="Percentage for validation split (e.g., 0.01 for 1%)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for shuffling and splitting")
    parser.add_argument("--header", type=int, default=None, help="Row number to use as header (e.g., 0) or None if no header.")
    parser.add_argument("--input_query_col_idx", type=int, default=0, help="0-based index of the query column in the input file.")
    parser.add_argument("--input_positive_col_idx", type=int, default=1, help="0-based index of the positive passage column in the input file.")


    args = parser.parse_args()

    logger.info(f"Loading contrastive data from {args.input_file}...")
    try:
        # Read specific columns using usecols based on index
        df = pd.read_csv(
            args.input_file,
            sep=args.delimiter,
            header=args.header,
            usecols=[args.input_query_col_idx, args.input_positive_col_idx],
            # Assign names immediately based on output desired
            names=[args.query_column_out, args.positive_column_out] if args.header is None else None,
            on_bad_lines='warn', # Warn about lines that couldn't be parsed
            encoding='utf-8' # Specify encoding
        )
        # If header was present, rename columns to desired output names
        if args.header is not None:
             # Get current column names based on index
             current_query_col_name = df.columns[args.input_query_col_idx]
             current_positive_col_name = df.columns[args.input_positive_col_idx]
             df.rename(columns={
                 current_query_col_name: args.query_column_out,
                 current_positive_col_name: args.positive_column_out
             }, inplace=True)
             # Keep only the renamed columns
             df = df[[args.query_column_out, args.positive_column_out]]


        logger.info(f"Loaded DataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Sample data:\n{df.head()}")

        # Data Cleaning
        initial_count = len(df)
        df.dropna(subset=[args.query_column_out, args.positive_column_out], inplace=True)
        df = df[df[args.query_column_out].astype(str).str.strip() != '']
        df = df[df[args.positive_column_out].astype(str).str.strip() != '']
        final_count = len(df)
        logger.info(f"Removed {initial_count - final_count} rows during cleaning (NaN or empty strings).")


    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return
    except Exception as e:
        logger.error(f"Error loading or processing data: {e}", exc_info=True)
        return

    if df.empty:
        logger.error("No valid data loaded after cleaning. Check file format, delimiter, column indices, and content.")
        return

    logger.info(f"Loaded {len(df)} valid query-positive pairs after cleaning.")

    # Split dataset
    logger.info(f"Splitting data: {(1-args.validation_split)*100:.1f}% train, {args.validation_split*100:.1f}% validation...")
    if len(df) < 2:
        logger.error("Not enough data to split into training and validation sets after cleaning. Need at least 2 examples.")
        return

    try:
        train_df, val_df = train_test_split(
            df,
            test_size=args.validation_split,
            random_state=args.random_seed,
            shuffle=True
        )
    except ValueError as e:
        logger.error(f"Error during train/test split: {e}. Check validation_split value and data size.")
        logger.warning("Using all data for training as split failed.")
        train_df = df
        val_df = pd.DataFrame(columns=[args.query_column_out, args.positive_column_out]) # Empty validation set


    # Save to disk
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, "train.csv")
        val_path = os.path.join(args.output_dir, "validation.csv")

        train_df.to_csv(train_path, index=False, encoding='utf-8')
        if not val_df.empty:
            val_df.to_csv(val_path, index=False, encoding='utf-8')
        else:
             # Create empty validation file with header
             with open(val_path, 'w', encoding='utf-8') as f:
                 f.write(f"{args.query_column_out},{args.positive_column_out}\n")


        logger.info(f"Saved {len(train_df)} training examples to {train_path}")
        logger.info(f"Saved {len(val_df)} validation examples to {val_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    main()
```

**5. Contrastive Fine-tuning Script (`run_contrastive_finetuning.py`)**

```python
# run_contrastive_finetuning.py

import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import datasets
import torch
from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.datasets import NoDuplicatesDataLoader # Important for MNRL
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SentenceEvaluator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which Sentence Transformer model we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Sentence Transformer model (e.g., output from MLM stage) or identifier from Hub."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store downloaded models"},
    )
    add_special_tokens: bool = field(
        default=False, metadata={"help": "Add [CLS] and [SEP] tokens. Usually False for ST, handled internally."}
    )
    # E5 specific prefixing - control whether to add "query: " and "passage: "
    prefix_query: Optional[str] = field(
        default=None, metadata={"help": "Prefix to add to queries (e.g., 'query: '). None means no prefix."}
    )
    prefix_passage: Optional[str] = field(
        default=None, metadata={"help": "Prefix to add to passages (e.g., 'passage: '). None means no prefix."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for contrastive training.
    """
    train_file: str = field(
        metadata={"help": "The input training data file (a CSV or JSONL file). Required."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (CSV or JSONL)."}
    )
    query_column: str = field(
        default="query",
        metadata={"help": "Name of the column containing queries."}
    )
    positive_column: str = field(
        default="positive",
        metadata={"help": "Name of the column containing positive passages."}
    )
    # Negative column is often not needed when using MultipleNegativesRankingLoss
    # negative_column: Optional[str] = field(default=None, metadata={"help": "Name of the column containing negative passages."})
    max_seq_length: Optional[int] = field(
        default=None, # Default to None, let ST handle or use model's default
        metadata={"help": "The maximum total input sequence length after tokenization. If None, uses model default."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing (Dataset loading)."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of evaluation examples."}
    )


@dataclass
class ContrastiveTrainingArguments:
    """
    Arguments pertaining to the training process itself using Sentence Transformers.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the final model and checkpoints will be written."},
    )
    num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=64, metadata={"help": "Batch size per GPU/CPU for training."})
    # Note: gradient_accumulation_steps is handled implicitly by batch size in ST's fit()
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Linear warmup over warmup_ratio*total_steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW optimizer."})
    save_strategy: str = field(default="epoch", metadata={"help": "When to save checkpoints ('no', 'epoch'). ST fit handles steps internally based on evaluation."})
    # save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."}) # Handled by evaluation_steps
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Limit the total amount of checkpoints saved."})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    evaluation_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy ('no', 'epoch'). Steps are possible via evaluation_steps."})
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Run evaluation every X steps. If None and strategy is epoch, evaluates each epoch."})
    use_amp: bool = field(default=False, metadata={"help": "Use Automatic Mixed Precision (requires PyTorch >= 1.6)."})
    # Loss specific arguments
    loss_scale: float = field(default=20.0, metadata={"help": "Scale factor for cosine similarity in MultipleNegativesRankingLoss (like temperature)."})


def main():
    # --- 1. Parse arguments ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ContrastiveTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 2. Setup logging ---
    # Logger is configured globally at the top
    logger.info(f"Contrastive Training Arguments {training_args}")
    logger.info(f"Model Arguments {model_args}")
    logger.info(f"Data Arguments {data_args}")

    # --- 3. Set Seed ---
    set_seed(training_args.seed)

    # --- 4. Load Model ---
    logger.info(f"Loading Sentence Transformer model: {model_args.model_name_or_path}")
    # This expects a path to a Sentence Transformer model directory
    # (e.g., './mlm_output_base/sentence-transformer-format') or a model name from the Hub
    try:
        model = SentenceTransformer(
            model_args.model_name_or_path,
            cache_folder=model_args.cache_dir,
            # device=training_args.device # ST usually handles device placement
        )
        logger.info(f"Model loaded successfully. Type: {type(model)}")
        # Log model card data if available
        if hasattr(model, 'model_card_data'):
             logger.info(f"Model card data: {model.model_card_data}")

    except Exception as e:
        logger.error(f"Failed to load Sentence Transformer model from {model_args.model_name_or_path}: {e}", exc_info=True)
        raise e

    # Set max sequence length if specified
    if data_args.max_seq_length is not None:
        original_max_len = model.get_max_seq_length()
        if original_max_len != data_args.max_seq_length:
            try:
                model.max_seq_length = data_args.max_seq_length
                logger.info(f"Set model max sequence length to: {data_args.max_seq_length} (was {original_max_len})")
            except Exception as e:
                 logger.warning(f"Could not set max_seq_length on the loaded model: {e}. Using model default: {original_max_len}")
        else:
             logger.info(f"Model max sequence length already matches specified value: {data_args.max_seq_length}")


    # --- 5. Load Data ---
    logger.info("Loading contrastive datasets...")
    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    else:
        logger.info("No validation file provided.")

    extension = data_args.train_file.split(".")[-1]
    if extension not in ["csv", "json", "jsonl"]:
        raise ValueError(f"Unsupported train file extension: {extension}. Must be csv, json, or jsonl")
    load_format = "json" if extension == "jsonl" else extension

    try:
        raw_datasets = load_dataset(
            load_format,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers, # Use multiple workers for loading if specified
            # streaming=True, # Consider for very large datasets, requires different handling below
        )
        logger.info(f"Raw datasets loaded: {raw_datasets}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        raise e

    # --- 6. Prepare Training Data ---
    logger.info("Preparing training examples (InputExample format)...")
    train_samples: List[InputExample] = []
    train_data = raw_datasets['train']
    if data_args.max_train_samples is not None:
        logger.info(f"Limiting training examples to {data_args.max_train_samples}")
        train_data = train_data.select(range(data_args.max_train_samples))

    skipped_count = 0
    for i, example in enumerate(train_data):
        try:
            query = example.get(data_args.query_column)
            positive = example.get(data_args.positive_column)

            # Ensure data are strings and not empty
            if isinstance(query, str) and query.strip() and isinstance(positive, str) and positive.strip():
                # Apply optional prefixes
                if model_args.prefix_query:
                    query = model_args.prefix_query + query
                if model_args.prefix_passage:
                    positive = model_args.prefix_passage + positive

                train_samples.append(InputExample(texts=[query, positive]))
            else:
                skipped_count += 1
                if skipped_count < 10: # Log first few skips
                    logger.warning(f"Skipping training example #{i} due to missing/invalid query or positive: Query='{query}', Positive='{positive}'")
                elif skipped_count == 10:
                    logger.warning("Further skipping warnings will be suppressed.")

        except Exception as e:
            logger.warning(f"Error processing training example #{i}: {example}. Error: {e}")
            skipped_count += 1

    logger.info(f"Created {len(train_samples)} training examples.")
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} training examples due to missing/invalid data.")
    if not train_samples:
        raise ValueError("No valid training examples were created. Check data format, column names, and content.")

    # Special dataloader for MultipleNegativesRankingLoss: Ensures no duplicate passages in a batch
    logger.info("Creating NoDuplicatesDataLoader for training...")
    train_dataloader = NoDuplicatesDataLoader(
        train_samples,
        batch_size=training_args.per_device_train_batch_size
    )
    logger.info(f"Train DataLoader created. Batch size: {training_args.per_device_train_batch_size}")

    # --- 7. Define Loss Function ---
    # MultipleNegativesRankingLoss is standard for contrastive learning with in-batch negatives (InfoNCE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model, scale=training_args.loss_scale)
    logger.info(f"Using MultipleNegativesRankingLoss with scale (temperature equivalent): {training_args.loss_scale}")

    # --- 8. Prepare Evaluation Data (Optional) ---
    evaluator: Optional[SentenceEvaluator] = None
    if data_args.validation_file and training_args.evaluation_strategy != "no":
        logger.info("Preparing evaluation data for InformationRetrievalEvaluator...")
        eval_data = raw_datasets['validation']
        if data_args.max_eval_samples is not None:
            logger.info(f"Limiting evaluation examples to {data_args.max_eval_samples}")
            eval_data = eval_data.select(range(data_args.max_eval_samples))

        # Structure needed for InformationRetrievalEvaluator
        queries: Dict[str, str] = {} # qid -> query_text
        corpus: Dict[str, str] = {} # docid -> doc_text (using positive passages as corpus)
        relevant_docs: Dict[str, List[str]] = {} # qid -> list[docids]

        qid_counter = 0
        docid_counter = 0
        query_to_qid: Dict[str, str] = {}
        passage_to_docid: Dict[str, str] = {}
        eval_skipped_count = 0

        for i, example in enumerate(eval_data):
            try:
                query_text = example.get(data_args.query_column)
                positive_text = example.get(data_args.positive_column)

                # Ensure data are strings and not empty
                if not (isinstance(query_text, str) and query_text.strip() and isinstance(positive_text, str) and positive_text.strip()):
                    eval_skipped_count += 1
                    if eval_skipped_count < 10:
                         logger.warning(f"Skipping eval example #{i} due to missing/invalid query or positive: Query='{query_text}', Positive='{positive_text}'")
                    elif eval_skipped_count == 10:
                         logger.warning("Further eval skipping warnings will be suppressed.")
                    continue

                # Apply optional prefixes CONSISTENTLY with training data
                if model_args.prefix_query:
                    query_text = model_args.prefix_query + query_text
                if model_args.prefix_passage:
                    positive_text = model_args.prefix_passage + positive_text


                # Assign unique IDs for queries
                if query_text not in query_to_qid:
                    qid = f"q_{qid_counter}"
                    queries[qid] = query_text
                    query_to_qid[query_text] = qid
                    relevant_docs[qid] = [] # Initialize list for relevant docs
                    qid_counter += 1
                else:
                    qid = query_to_qid[query_text]

                # Assign unique IDs for passages (corpus)
                if positive_text not in passage_to_docid:
                    docid = f"doc_{docid_counter}"
                    corpus[docid] = positive_text
                    passage_to_docid[positive_text] = docid
                    docid_counter += 1
                else:
                    docid = passage_to_docid[positive_text]

                # Add the current positive passage's docid to the relevant docs for the query
                if docid not in relevant_docs[qid]:
                    relevant_docs[qid].append(docid)

            except Exception as e:
                logger.warning(f"Error processing evaluation example #{i}: {example}. Error: {e}")
                eval_skipped_count += 1


        if eval_skipped_count > 0:
             logger.warning(f"Skipped {eval_skipped_count} evaluation examples due to missing/invalid data.")

        if queries and corpus and relevant_docs:
            logger.info(f"Creating InformationRetrievalEvaluator with {len(queries)} queries, {len(corpus)} corpus docs.")
            evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                main_score_function='cos_sim', # Use cosine similarity for scoring
                # score_functions={'cos_sim': lambda q, p: torch.nn.functional.cosine_similarity(q, p, dim=-1)}, # Redundant if main_score_function is set
                main_score='map@10', # Main metric to track (e.g., MAP@10, NDCG@10) - choose one
                name="validation",
                show_progress_bar=True,
                # Adjust batch size for evaluation if needed (can often be larger)
                batch_size=training_args.per_device_train_batch_size * 2,
                precision_recall_at_k=[1, 5, 10, 100], # K values for precision/recall
                map_at_k=[10, 100], # K values for MAP
                ndcg_at_k=[10, 100] # K values for NDCG
            )
            logger.info("InformationRetrievalEvaluator created.")
        else:
            logger.warning("Could not create evaluator due to missing queries, corpus, or relevant docs after processing validation data.")
            if training_args.evaluation_strategy != "no":
                 logger.warning("Evaluation strategy is set but no evaluator could be created. Disabling evaluation.")
                 training_args.evaluation_strategy = "no"

    # --- 9. Configure Training Steps and Warmup ---
    # Calculate total steps and warmup steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps) # ST handles grad accum internally via batch size
    if training_args.num_train_epochs <= 0: # Handle case where epochs is not set (e.g., if max_steps is used, though less common in ST fit)
         # Need a way to estimate total steps if epochs isn't the driver
         # For now, assume epochs is the primary driver
         logger.warning("num_train_epochs is not positive. Warmup calculation might be incorrect if max_steps is intended.")
         total_training_steps = 1 # Placeholder to avoid division by zero
    else:
        total_training_steps = num_update_steps_per_epoch * training_args.num_train_epochs

    warmup_steps = math.ceil(total_training_steps * training_args.warmup_ratio)
    logger.info(f"Total optimization steps: {total_training_steps} (estimated)")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Determine evaluation steps based on strategy
    evaluation_steps = 0
    if training_args.evaluation_strategy == "epoch":
        evaluation_steps = num_update_steps_per_epoch # Evaluate once per epoch
        logger.info(f"Evaluation strategy: 'epoch'. Evaluating every {evaluation_steps} steps.")
    elif training_args.evaluation_strategy == "steps":
        if training_args.eval_steps is None or training_args.eval_steps <= 0:
             evaluation_steps = num_update_steps_per_epoch # Default to once per epoch if steps not specified
             logger.warning(f"Evaluation strategy 'steps' chosen but eval_steps not set. Defaulting to eval per epoch ({evaluation_steps} steps).")
        else:
             evaluation_steps = training_args.eval_steps
             logger.info(f"Evaluation strategy: 'steps'. Evaluating every {evaluation_steps} steps.")
    else: # 'no'
         logger.info("Evaluation strategy: 'no'. No evaluation during training.")


    # --- 10. Train the model ---
    logger.info("*** Starting Contrastive Fine-tuning ***")
    logger.info(f"  Num examples = {len(train_samples)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Training batch size = {training_args.per_device_train_batch_size}")
    logger.info(f"  Learning Rate = {training_args.learning_rate}")
    logger.info(f"  Warmup Steps = {warmup_steps}")
    logger.info(f"  Use AMP = {training_args.use_amp}")
    logger.info(f"  Output Directory = {training_args.output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=training_args.num_train_epochs,
            evaluation_steps=evaluation_steps if evaluator else 0, # Only set if evaluator exists
            warmup_steps=warmup_steps,
            optimizer_params={'lr': training_args.learning_rate},
            weight_decay=training_args.weight_decay,
            output_path=training_args.output_dir, # Saves the final model here
            save_best_model=evaluator is not None, # Save best model based on evaluator score if evaluator exists
            checkpoint_path=os.path.join(training_args.output_dir, "checkpoints"), # Subdir for checkpoints
            checkpoint_save_total_limit=training_args.save_total_limit if training_args.save_strategy != "no" else 0,
            show_progress_bar=True,
            use_amp=training_args.use_amp
        )
        logger.info("Training finished successfully.")

    except Exception as e:
        logger.error(f"Error during model.fit(): {e}", exc_info=True)
        raise e

    # --- 11. Final Save (Optional, fit usually saves the best/last model) ---
    # You might want an explicit save if save_best_model=False or just to be sure
    final_save_path = os.path.join(training_args.output_dir, "final_model")
    logger.info(f"Saving final model explicitly to {final_save_path} (might overwrite best model if different)")
    model.save(final_save_path)

    logger.info(f"Contrastive fine-tuning finished. Final model saved in {training_args.output_dir} (and potentially {final_save_path}).")


if __name__ == "__main__":
    main()
```

**6. Example Usage Script (`run_pipeline.sh`)**

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---

# Stage 0: Data Paths
# Assumes you have:
# - A text file (or multiple) with one sentence/document per line for MLM
# - A TSV/CSV file with query<tab>positive_passage for contrastive learning
MLM_RAW_TEXT_FILES="./my_corpus.txt" # Path to your raw text file(s) for MLM
CONTRASTIVE_RAW_FILE="./query_positive_pairs.tsv" # Path to query<tab>positive file

MLM_PREPARED_DATA_DIR="./data_prepared/mlm_base"
CONTRASTIVE_PREPARED_DATA_DIR="./data_prepared/contrastive_base"

# Stage 1: MLM Pre-training Config
# STARTING FROM BERT-BASE-UNCASED
BASE_MODEL="bert-base-uncased"
MLM_OUTPUT_DIR="./output/mlm_from_bert_base"
MLM_EPOCHS=3
MLM_BATCH_SIZE=16 # Adjust based on GPU memory
MLM_LR=5e-5
MLM_MAX_LENGTH=512 # Sequence length for MLM

# Stage 2: Contrastive Fine-tuning Config
# Input model is the output of Stage 1's Sentence Transformer conversion
CONTRASTIVE_INPUT_MODEL="${MLM_OUTPUT_DIR}/sentence-transformer-format"
CONTRASTIVE_OUTPUT_DIR="./output/e5_finetuned_from_bert_base"
CONTRASTIVE_EPOCHS=1 # Often 1 epoch is enough for contrastive fine-tuning
CONTRASTIVE_BATCH_SIZE=64 # Can often be larger for contrastive
CONTRASTIVE_LR=2e-5
CONTRASTIVE_MAX_LENGTH=512 # Should match MLM stage or model's capability
# Optional E5-style prefixes (set to "query: " and "passage: " if desired)
QUERY_PREFIX="query: " # Example prefix, set to "" or remove arg for none
PASSAGE_PREFIX="passage: " # Example prefix, set to "" or remove arg for none

# --- Execution ---

echo "===== Stage 0a: Preparing MLM Data ====="
python prepare_mlm_data.py \
    --input_files $MLM_RAW_TEXT_FILES \
    --output_dir $MLM_PREPARED_DATA_DIR \
    --validation_split 0.01 \
    --text_column text \
    --random_seed 42

echo "===== Stage 0b: Preparing Contrastive Data ====="
python prepare_contrastive_data.py \
    --input_file $CONTRASTIVE_RAW_FILE \
    --output_dir $CONTRASTIVE_PREPARED_DATA_DIR \
    --delimiter "\t" \
    --input_query_col_idx 0 \
    --input_positive_col_idx 1 \
    --query_column_out query \
    --positive_column_out positive \
    --validation_split 0.01 \
    --random_seed 42

echo "===== Stage 1: Running MLM Pre-training from ${BASE_MODEL} ====="
# Use accelerate for multi-gpu or deepspeed support if needed
# Adjust --num_processes based on your GPU count for multi-GPU
accelerate launch run_mlm_pretraining.py \
    --model_name_or_path $BASE_MODEL \
    --train_file "${MLM_PREPARED_DATA_DIR}/train.csv" \
    --validation_file "${MLM_PREPARED_DATA_DIR}/validation.csv" \
    --text_column_name text \
    --do_train \
    --do_eval \
    --output_dir $MLM_OUTPUT_DIR \
    --overwrite_output_dir \
    --num_train_epochs $MLM_EPOCHS \
    --per_device_train_batch_size $MLM_BATCH_SIZE \
    --per_device_eval_batch_size $(($MLM_BATCH_SIZE * 2)) \
    --gradient_accumulation_steps 2 \
    --learning_rate $MLM_LR \
    --max_seq_length $MLM_MAX_LENGTH \
    --line_by_line False \
    --pad_to_max_length False \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --save_total_limit 2 \
    --logging_steps 100 \
    --fp16 # Use FP16 if available

echo "===== Stage 2: Running Contrastive Fine-tuning ====="
# Check if the input model directory exists
if [ ! -d "$CONTRASTIVE_INPUT_MODEL" ]; then
    echo "ERROR: Input model for contrastive stage not found at $CONTRASTIVE_INPUT_MODEL"
    echo "MLM pre-training or Sentence Transformer conversion might have failed."
    exit 1
fi

# Contrastive stage doesn't typically use accelerate directly, ST handles multi-GPU internally
python run_contrastive_finetuning.py \
    --model_name_or_path $CONTRASTIVE_INPUT_MODEL \
    --train_file "${CONTRASTIVE_PREPARED_DATA_DIR}/train.csv" \
    --validation_file "${CONTRASTIVE_PREPARED_DATA_DIR}/validation.csv" \
    --query_column query \
    --positive_column positive \
    --output_dir $CONTRASTIVE_OUTPUT_DIR \
    --num_train_epochs $CONTRASTIVE_EPOCHS \
    --per_device_train_batch_size $CONTRASTIVE_BATCH_SIZE \
    --learning_rate $CONTRASTIVE_LR \
    --max_seq_length $CONTRASTIVE_MAX_LENGTH \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --warmup_ratio 0.1 \
    --use_amp True \
    --prefix_query "$QUERY_PREFIX" \
    --prefix_passage "$PASSAGE_PREFIX" \
    --seed 42

echo "===== Pipeline Finished ====="
echo "MLM Pre-trained HF model: ${MLM_OUTPUT_DIR}"
echo "MLM Pre-trained ST model: ${CONTRASTIVE_INPUT_MODEL}"
echo "Contrastive Fine-tuned ST model: ${CONTRASTIVE_OUTPUT_DIR}"
```

This provides the complete code for all steps, starting from `bert-base-uncased`, performing MLM pre-training, converting to Sentence Transformer format, and then performing contrastive fine-tuning. Remember to adjust file paths, batch sizes, and other parameters according to your specific data and hardware resources.