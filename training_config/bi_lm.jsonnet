// Adapted from https://github.com/allenai/allennlp-models/blob/main/training_config/lm/bidirectional_language_model.jsonnet

local TRAIN = std.extVar("TRAIN");
local CUDA = std.parseInt(std.extVar("CUDA"));

local NUM_GPUS = 1;
local NUM_GRAD_ACC = 4;
local BATCH_SIZE = 512 / NUM_GPUS / NUM_GRAD_ACC;
local N_EPOCHS = 3;

local D_MODEL = 216;
local D_FF = 256;
local N_LAYERS = 3;

local BASE_READER = {
        "type": "simple_language_modeling",
        "tokenizer": {
          "type": "just_spaces"
        },
        "token_indexers": {
          "tokens": {
            "type": "single_id"
          },
        },
        "max_sequence_length": 400,
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"],
};

local BASE_LOADER = {
  "max_instances_in_memory": BATCH_SIZE * 100,
  "batch_sampler": {
    "type": "bucket",
    "batch_size": BATCH_SIZE,
  }
};

{
  "dataset_reader": BASE_READER,
  // Note: We don't set a validation_data_path because the softmax is only
  // sampled during training. Not sampling on GPUs results in a certain OOM
  // given our large vocabulary. We'll need to evaluate against the test set
  // (when we'll want a full softmax) with the CPU.
  "train_data_path": TRAIN,

//   "vocabulary": {
//       // Use a prespecified vocabulary for efficiency.
//       "type": "from_files",
//       "directory": std.extVar("BIDIRECTIONAL_LM_VOCAB_PATH"),
//       // Plausible config for generating the vocabulary.
//       // "tokens_to_add": {
//       //     "tokens": ["<S>", "</S>"],
//       //     "token_characters": ["<>/S"]
//       // },
//       // "min_count": {"tokens": 3}
//   },
  "model": {
    "type": "language_model",
    "bidirectional": true,
    // "num_samples": 8192,  # In our case, the vocabulary is very small.
    # Sparse embeddings don't work with DistributedDataParallel.
    "sparse_embeddings": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": D_MODEL
        }
      }
    },
    // TODO(brendanr): Consider the following.
    // remove_bos_eos: true,
    // Applies to the contextualized embeddings.
    "dropout": 0.1,
    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": D_MODEL,
        "hidden_dim": D_FF,
        "num_layers": N_LAYERS,
        "dropout": 0.1,
        "input_dropout": 0.1
    }
  },
  "data_loader": BASE_LOADER,
  // "distributed": {
  //   "cuda_devices": if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
  // },
  "trainer": {
    "num_epochs": N_EPOCHS,
    "optimizer": {
      // The gradient accumulators in Adam for the running stdev and mean for
      // words not used in the sampled softmax would be decayed to zero with the
      // standard "adam" optimizer.
      "type": "dense_sparse_adam"
    },
    // TODO(brendanr): Needed with transformer too?
    // "grad_norm": 10.0,
    "learning_rate_scheduler": {
      "type": "noam",
      // See https://github.com/allenai/calypso/blob/master/calypso/train.py#L401
      "model_size": 512,
      // See https://github.com/allenai/calypso/blob/master/bin/train_transformer_lm1b.py#L51.
      // Adjusted based on our sample size relative to Calypso's.
      "warmup_steps": 6000
    },
    "num_gradient_accumulation_steps": NUM_GRAD_ACC,
    // "use_amp": true,
    "device": CUDA
  }
}
