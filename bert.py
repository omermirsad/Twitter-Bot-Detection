Okay, let's break this down into two main parts:

1.  **MLM Pre-training Script:** This is the refined version of the script we've been working on. It performs Masked Language Modeling and includes the conversion to the Sentence Transformer format at the end.
2.  **Contrastive Fine-tuning Script:** This is a *new* script specifically for the E5-style contrastive fine-tuning step, using query-passage pairs and the `sentence-transformers` library.

---

**Part 1: MLM Pre-training Script (`run_mlm_pretraining.py`)**

This script takes a base model (like `bert-base-uncased` or `intfloat/e5-small-v2`) and performs MLM pre-training on your custom text corpus.

```python
# run_mlm_pretraining.py

# requirements.txt
# ------------------------
# torch>=2.0.0
# transformers>=4.30.0
# datasets>=2.12.0
# accelerate>=0.20.0
# deepspeed>=0.9.5 # Optional, for large scale
# sentence-transformers>=2.2.2
# tensorboard>=2.13.0 # Optional, for logging
# wandb>=0.15.0 # Optional, for logging
# evaluate # For metrics calculation

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import datasets
import torch
import transformers
from datasets import load_dataset
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

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0")
require_version("datasets>=2.12.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    token: str = field(
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
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 2. Setup logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.should_log else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # --- 3. Detect Checkpoints ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # ... (checkpoint detection logic as before) ...

    # --- 4. Set Seed ---
    set_seed(training_args.seed)

    # --- 5. Load Data ---
    # ... (data loading logic using dataset_name or train/validation_file as before) ...
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        if "validation" not in raw_datasets.keys():
            logger.info(f"Validation split not found. Splitting train set: {100-data_args.validation_split_percentage}% train, {data_args.validation_split_percentage}% validation.")
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl"]:
                 raise ValueError("Train file extension must be csv, json, jsonl or txt")
        else:
             raise ValueError("You must specify a train_file")

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl"]:
                 raise ValueError("Validation file extension must be csv, json, jsonl or txt")
        else:
            extension = data_args.train_file.split(".")[-1] # Use train file extension if validation is missing

        # Adjust loading based on extension
        load_format = extension
        if extension == "txt":
            load_format = "text" # datasets library uses "text" loader for txt files
        elif extension == "jsonl":
             load_format = "json"

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
            # Shuffle before splitting
            split_datasets = raw_datasets["train"].train_test_split(
                test_size=data_args.validation_split_percentage / 100.0,
                seed=training_args.seed,
                load_from_cache_file=not data_args.overwrite_cache
            )
            raw_datasets["train"] = split_datasets["train"]
            raw_datasets["validation"] = split_datasets["test"]

    # --- 6. Load Model & Tokenizer ---
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    # ... (config loading logic as before) ...
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # Handle case where model_args.model_type needs to be defined for scratch training
        raise ValueError("Must provide model_name_or_path or config_name")


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    # ... (tokenizer loading logic as before) ...
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("Must provide model_name_or_path or tokenizer_name")


    logger.info(f"Loading model {model_args.model_name_or_path} for Masked LM pre-training.")
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    model.resize_token_embeddings(len(tokenizer))

    # --- 7. Preprocess Data ---
    # ... (preprocessing logic using tokenize_function and group_texts as before) ...
    # Determine text column name
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    elif training_args.do_eval:
        column_names = list(raw_datasets["validation"].features)
    else:
         # Handle case where neither do_train nor do_eval is set but preprocessing is needed
         # Or default to a common case if applicable
         if "train" in raw_datasets:
             column_names = list(raw_datasets["train"].features)
         elif "validation" in raw_datasets:
             column_names = list(raw_datasets["validation"].features)
         else:
             raise ValueError("Cannot determine dataset columns without a train or validation split.")


    if data_args.text_column_name is None or data_args.text_column_name not in column_names:
         logger.warning(f"--text_column_name '{data_args.text_column_name}' not found in dataset columns: {column_names}. Trying to use 'text'.")
         if 'text' in column_names:
             text_column_name = 'text'
         else:
             # If 'text' isn't found either, raise error or pick the first string column?
             # For now, raise error for clarity.
             raise ValueError(f"Cannot find text column. Specify --text_column_name among {column_names}")
    else:
        text_column_name = data_args.text_column_name
    logger.info(f"Using '{text_column_name}' as the text column.")


    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024: # Hard cap
            logger.warning(f"Tokenizer max length is {max_seq_length}. Setting max_seq_length to 1024.")
            max_seq_length = 1024
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            return_special_tokens_mask=True # Needed for MLM collator
        )

    # Grouping function (for line_by_line=False)
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, adjust as needed
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        # Add attention_mask back if it was removed during concatenation
        result["attention_mask"] = [[1] * len(result["input_ids"][i]) for i in range(len(result["input_ids"]))]
        return result

    with training_args.main_process_first(desc="dataset map tokenization"):
        if data_args.line_by_line:
             # Tokenize each line separately
             padding = "max_length" if data_args.pad_to_max_length else False
             def tokenize_line_by_line(examples):
                 lines = [line for line in examples[text_column_name] if len(line.strip()) > 0]
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
                 remove_columns=[text_column_name],
                 load_from_cache_file=not data_args.overwrite_cache,
                 desc="Running tokenizer on dataset line_by_line",
             )
        else:
            # Tokenize, then group
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, # Remove original columns
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )


    # --- 8. Prepare Datasets for Trainer ---
    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # --- 9. Data Collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )
    logger.info(f"Using DataCollatorForLanguageModeling with mlm_probability={data_args.mlm_probability}")

    # --- 10. Initialize Trainer ---
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

    # --- 11. Training ---
    if training_args.do_train:
        logger.info("*** Starting MLM Pre-training ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # Save the final HF model
        trainer.save_model()

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

    # --- 12. Evaluation ---
    if training_args.do_eval:
        logger.info("*** Evaluate MLM Performance ***")
        metrics = trainer.evaluate()

        if eval_dataset:
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # --- 13. Convert to Sentence Transformer format ---
    # This step takes the MLM-pretrained model and adds a pooling layer.
    # It makes the model directly usable for embedding generation via the sentence-transformers library.
    # NOTE: This model has *only* seen MLM pre-training, not the contrastive fine-tuning of official E5.
    final_model_path = training_args.output_dir
    if training_args.load_best_model_at_end and training_args.metric_for_best_model and trainer.state.best_model_checkpoint:
        final_model_path = trainer.state.best_model_checkpoint
        logger.info(f"Loading best model from {final_model_path} for Sentence Transformer conversion.")
    elif not training_args.do_train:
         # If only evaluating, use the input model path if no best checkpoint is found
         final_model_path = model_args.model_name_or_path
         logger.info(f"No training performed or best model found, using {final_model_path} for potential conversion.")


    # Only convert if a model exists at the path (either trained or loaded best)
    if os.path.exists(os.path.join(final_model_path, "pytorch_model.bin")) or \
       os.path.exists(os.path.join(final_model_path, "model.safetensors")):
        logger.info("Converting MLM-pretrained model to Sentence Transformers format...")
        try:
            from sentence_transformers import SentenceTransformer, models

            logger.info(f"Loading base transformer model from: {final_model_path}")
            word_embedding_model = models.Transformer(final_model_path)

            # Add a pooling layer (Mean Pooling is common, CLS is also used e.g., in E5 fine-tuning)
            pooling_model = models.Pooling(
                word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,  # Use mean pooling
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            logger.info("Added Mean Pooling layer.")

            st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            st_output_path = os.path.join(training_args.output_dir, "sentence-transformer-format")
            os.makedirs(st_output_path, exist_ok=True)
            st_model.save(st_output_path)
            logger.info(f"Model saved in Sentence Transformers format at: {st_output_path}")
            logger.info("This model can now be used as input for contrastive fine-tuning or directly for embeddings.")

        except ImportError:
            logger.warning("`sentence-transformers` library not found. Skipping conversion.")
            logger.warning("Install it with `pip install sentence-transformers` to enable this feature.")
        except Exception as e:
            logger.error(f"Error during Sentence Transformers conversion: {e}", exc_info=True)
    else:
         logger.warning(f"No model found at {final_model_path}. Skipping Sentence Transformer conversion.")


    # --- 14. Push to Hub / Model Card ---
    # ... (optional push_to_hub and create_model_card logic as before) ...


if __name__ == "__main__":
    main()
```

---

**Part 2: Contrastive Fine-tuning Script (`run_contrastive_finetuning.py`)**

This script takes a Sentence Transformer model (like the output from Part 1, or `intfloat/e5-base`) and fine-tunes it using contrastive learning on query-positive pairs. It uses the `sentence-transformers` library's training utilities.

```python
# run_contrastive_finetuning.py

# requirements.txt
# ------------------------
# torch>=2.0.0
# transformers>=4.30.0 # For tokenizer and base model if needed indirectly
# datasets>=2.12.0
# accelerate>=0.20.0 # Potentially needed by ST backend
# sentence-transformers>=2.2.2
# tensorboard>=2.13.0 # Optional, for logging
# wandb>=0.15.0 # Optional, for logging

import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import datasets
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.datasets import NoDuplicatesDataLoader # Important for MNRL
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which Sentence Transformer model we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Sentence Transformer model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # Add other model args if needed, e.g., pooling mode override, but usually ST handles this.


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for contrastive training.
    """
    train_file: str = field(
        metadata={"help": "The input training data file (a CSV or JSONL file)."}
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
        default=512, # Match the underlying transformer's max length
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of evaluation examples."}
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to the training process itself.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=64, metadata={"help": "Batch size per GPU/CPU for training."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Linear warmup over warmup_ratio*total_steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW optimizer."})
    save_strategy: str = field(default="epoch", metadata={"help": "Checkpoint saving strategy ('no', 'epoch', 'steps')."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Limit the total amount of checkpoints."})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization."})
    evaluation_strategy: str = field(default="no", metadata={"help": "Evaluation strategy ('no', 'epoch', 'steps')."})
    eval_steps: Optional[int] = field(default=None, metadata={"help": "Run evaluation every X steps."})
    # Add other relevant args: logging_steps, fp16, etc.


def main():
    # --- 1. Parse arguments ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 2. Setup logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # --- 3. Set Seed ---
    set_seed(training_args.seed)

    # --- 4. Load Model ---
    logger.info(f"Loading Sentence Transformer model: {model_args.model_name_or_path}")
    # If the MLM script saved the ST model, the path would be like './mlm_output/sentence-transformer-format'
    model = SentenceTransformer(model_args.model_name_or_path, cache_folder=model_args.cache_dir)

    # Optionally set max sequence length (might be already configured in the loaded model)
    if data_args.max_seq_length is not None and hasattr(model, 'max_seq_length'):
        logger.info(f"Setting model max sequence length to: {data_args.max_seq_length}")
        model.max_seq_length = data_args.max_seq_length
    elif data_args.max_seq_length is not None:
         logger.warning(f"Could not set max_seq_length on the loaded model. Ensure the base transformer config is correct.")


    # --- 5. Load Data ---
    logger.info("Loading data...")
    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file

    extension = data_args.train_file.split(".")[-1]
    if extension not in ["csv", "json", "jsonl"]:
        raise ValueError("Data file extension must be csv, json, or jsonl")
    load_format = "json" if extension == "jsonl" else extension

    raw_datasets = load_dataset(
        load_format,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        # Use streaming if dataset is huge? Might need adjustments.
    )

    # --- 6. Prepare Training Data ---
    logger.info("Preparing training examples...")
    train_samples = []
    train_data = raw_datasets['train']
    if data_args.max_train_samples is not None:
        train_data = train_data.select(range(data_args.max_train_samples))

    for example in train_data:
        query = example.get(data_args.query_column)
        positive = example.get(data_args.positive_column)
        if query and positive:
            # E5 prepends "query: " and "passage: " - do this if your base model expects it
            # or if you want to follow the E5 format strictly.
            # query = "query: " + query
            # positive = "passage: " + positive
            train_samples.append(InputExample(texts=[query, positive]))
        else:
            logger.warning(f"Skipping example due to missing query/positive: {example}")

    logger.info(f"Created {len(train_samples)} training examples.")
    if not train_samples:
        raise ValueError("No valid training examples found. Check data format and column names.")

    # Special dataloader for MultipleNegativesRankingLoss: It ensures no duplicate passages in a batch
    train_dataloader = NoDuplicatesDataLoader(
        train_samples,
        batch_size=training_args.per_device_train_batch_size
    )

    # --- 7. Define Loss Function ---
    # MultipleNegativesRankingLoss is standard for contrastive learning with in-batch negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    logger.info("Using MultipleNegativesRankingLoss (InfoNCE with in-batch negatives).")

    # --- 8. Prepare Evaluation Data (Optional) ---
    evaluator = None
    if data_args.validation_file and training_args.evaluation_strategy != "no":
        logger.info("Preparing evaluation data...")
        eval_data = raw_datasets['validation']
        if data_args.max_eval_samples is not None:
            eval_data = eval_data.select(range(data_args.max_eval_samples))

        queries = {} # qid -> query_text
        corpus = {} # docid -> doc_text (using positive passages as corpus)
        relevant_docs = {} # qid -> set(docids)

        qid_counter = 0
        docid_counter = 0
        query_to_qid = {}
        passage_to_docid = {}

        for example in eval_data:
            query_text = example.get(data_args.query_column)
            positive_text = example.get(data_args.positive_column)

            if not (query_text and positive_text):
                logger.warning(f"Skipping eval example due to missing query/positive: {example}")
                continue

            # Assign unique IDs
            if query_text not in query_to_qid:
                qid = f"q_{qid_counter}"
                queries[qid] = query_text
                query_to_qid[query_text] = qid
                relevant_docs[qid] = set()
                qid_counter += 1
            else:
                qid = query_to_qid[query_text]

            if positive_text not in passage_to_docid:
                docid = f"doc_{docid_counter}"
                corpus[docid] = positive_text
                passage_to_docid[positive_text] = docid
                docid_counter += 1
            else:
                docid = passage_to_docid[positive_text]

            relevant_docs[qid].add(docid)

        if queries and corpus and relevant_docs:
            logger.info(f"Created evaluator with {len(queries)} queries, {len(corpus)} corpus docs.")
            evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                main_score_function='cos_sim', # Use cosine similarity
                score_functions={'cos_sim': lambda q, p: torch.nn.functional.cosine_similarity(q, p, dim=-1)},
                main_score='map', # Main metric to track (e.g., MAP@K)
                name="validation",
                show_progress_bar=True,
                batch_size=training_args.per_device_train_batch_size * 2 # Often can use larger batch for eval
            )
        else:
            logger.warning("Could not create evaluator due to missing queries, corpus, or relevant docs.")


    # --- 9. Configure Training ---
    warmup_steps = math.ceil(len(train_dataloader) * training_args.num_train_epochs / training_args.gradient_accumulation_steps * training_args.warmup_ratio)
    logger.info(f"Warmup steps: {warmup_steps}")

    # --- 10. Train the model ---
    logger.info("*** Starting Contrastive Fine-tuning ***")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=training_args.num_train_epochs,
        evaluation_steps=training_args.eval_steps if training_args.evaluation_strategy == "steps" else len(train_dataloader) // 2, # Sensible default if eval per epoch
        warmup_steps=warmup_steps,
        optimizer_params={'lr': training_args.learning_rate},
        weight_decay=training_args.weight_decay,
        output_path=training_args.output_dir,
        save_best_model=evaluator is not None, # Save best model based on evaluator score
        checkpoint_save_steps=training_args.save_steps if training_args.save_strategy == "steps" else None,
        checkpoint_path=training_args.output_dir + "/checkpoints",
        checkpoint_save_total_limit=training_args.save_total_limit if training_args.save_strategy != "no" else 0,
        show_progress_bar=True,
        # use_amp=training_args.fp16 # Check sentence-transformers documentation for AMP/FP16 usage
    )

    # --- 11. Save Final Model (Optional, fit already saves) ---
    # model.save(training_args.output_dir) # fit usually saves the best/last model already
    logger.info(f"Contrastive fine-tuning finished. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
```

---

**Part 3: Data Preparation Scripts**

**A. For MLM Pre-training (`prepare_mlm_data.py`)**
(This is similar to the previous `prepare_data.py`, ensuring a single text column)

```python
# prepare_mlm_data.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    all_texts = []
    print("Loading text files for MLM...")
    for file_path in args.input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                all_texts.extend(texts)
                print(f"  Loaded {len(texts)} lines from {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}, skipping.")

    if not all_texts:
        print("Error: No text data loaded. Exiting.")
        return

    print(f"Total loaded text examples: {len(all_texts)}")
    df = pd.DataFrame(all_texts, columns=[args.text_column])

    print(f"Splitting data...")
    train_df, val_df = train_test_split(
        df, test_size=args.validation_split, random_state=args.random_seed, shuffle=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "validation.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')

    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")

if __name__ == "__main__":
    main()
```

**B. For Contrastive Fine-tuning (`prepare_contrastive_data.py`)**
(This assumes you have data in a format like `query \t positive_passage`)

```python
# prepare_contrastive_data.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Contrastive fine-tuning (query, positive columns)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file (e.g., TSV with query<tab>positive_passage)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save train.csv and validation.csv")
    parser.add_argument("--delimiter", type=str, default="\t", help="Delimiter for input file")
    parser.add_argument("--query_column_out", type=str, default="query", help="Output column name for queries")
    parser.add_argument("--positive_column_out", type=str, default="positive", help="Output column name for positives")
    parser.add_argument("--validation_split", type=float, default=0.01, # Can be small if you have a dedicated large eval set
                        help="Percentage for validation split (e.g., 0.01 for 1%)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--header", action="store_true", help="Set if the input file has a header row")

    args = parser.parse_args()

    print(f"Loading contrastive data from {args.input_file}...")
    try:
        header = 0 if args.header else None
        # Assuming two columns: query, positive
        df = pd.read_csv(args.input_file, sep=args.delimiter, header=header, names=[args.query_column_out, args.positive_column_out], on_bad_lines='warn')
        df.dropna(inplace=True) # Remove rows with missing values
        df = df[df[args.query_column_out].astype(str).str.strip() != ''] # Remove empty queries
        df = df[df[args.positive_column_out].astype(str).str.strip() != ''] # Remove empty positives

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("Error: No valid data loaded after cleaning. Check file format and delimiter.")
        return

    print(f"Loaded {len(df)} valid query-positive pairs.")

    print(f"Splitting data...")
    train_df, val_df = train_test_split(
        df, test_size=args.validation_split, random_state=args.random_seed, shuffle=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "validation.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')

    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")

if __name__ == "__main__":
    main()
```

---

**Part 4: Requirements (`requirements.txt`)**

```
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

---

**How to Use:**

1.  **Prepare MLM Data:**
    ```bash
    python prepare_mlm_data.py \
        --input_files path/to/your/corpus1.txt path/to/your/corpus2.txt \
        --output_dir ./mlm_data
    ```
2.  **Run MLM Pre-training:**
    ```bash
    # Example using bert-small as base
    MODEL_BASE="google/bert_uncased_L-4_H-512_A-8"
    MLM_OUTPUT_DIR="./mlm_output_small"

    accelerate launch run_mlm_pretraining.py \
        --model_name_or_path $MODEL_BASE \
        --train_file ./mlm_data/train.csv \
        --validation_file ./mlm_data/validation.csv \
        --text_column_name text \
        --do_train \
        --do_eval \
        --output_dir $MLM_OUTPUT_DIR \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --max_seq_length 512 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --load_best_model_at_end True \
        --fp16 # If GPU supports it
    ```
    *   The MLM-pretrained HF model is in `$MLM_OUTPUT_DIR`.
    *   The Sentence Transformer compatible model is in `$MLM_OUTPUT_DIR/sentence-transformer-format`.

3.  **Prepare Contrastive Data:** (Assuming `query<tab>positive` format)
    ```bash
    python prepare_contrastive_data.py \
        --input_file path/to/your/query_positive_pairs.tsv \
        --output_dir ./contrastive_data \
        --delimiter "\t"
    ```
4.  **Run Contrastive Fine-tuning:**
    ```bash
    # Use the output from the MLM step
    CONTRASTIVE_INPUT_MODEL="$MLM_OUTPUT_DIR/sentence-transformer-format"
    CONTRASTIVE_OUTPUT_DIR="./e5_finetuned_small"

    python run_contrastive_finetuning.py \
        --model_name_or_path $CONTRASTIVE_INPUT_MODEL \
        --train_file ./contrastive_data/train.csv \
        --validation_file ./contrastive_data/validation.csv \
        --query_column query \
        --positive_column positive \
        --output_dir $CONTRASTIVE_OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size 64 \
        --learning_rate 2e-5 \
        --max_seq_length 512 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --save_total_limit 1 \
        --warmup_ratio 0.1
        # Add --fp16 if supported and desired (check ST docs)
    ```
    *   The final, contrastively fine-tuned Sentence Transformer model will be in `$CONTRASTIVE_OUTPUT_DIR`.

This provides a complete pipeline from raw text to an MLM-pretrained model, and then to a contrastively fine-tuned E5-style embedding model. Remember to adjust paths, model names, batch sizes, and other hyperparameters based on your specific data and hardware.