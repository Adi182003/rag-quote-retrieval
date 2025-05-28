---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: “people aren't born good or bad. maybe they're born with tendencies
    either way, but its the way you live your life that matters.”
  sentences:
  - “you see things; you say, 'why?' but i dream things that never were; and i say
    'why not?”
  - “people aren't born good or bad. maybe they're born with tendencies either way,
    but its the way you live your life that matters.”
  - “take responsibility of your own happiness, never put it in other peopleâ€™s hands.”
- source_sentence: “the reason it hurts so much to separate is because our souls are
    connected.”
  sentences:
  - “the reason it hurts so much to separate is because our souls are connected.”
  - “the only thing necessary for the triumph of evil is for good men to do nothing.”
  - “and i thought about how many people have loved those songs. and how many people
    got through a lot of bad times because of those songs. and how many people enjoyed
    good times with those songs. and how much those songs really mean. i think it
    would be great to have written one of those songs. i bet if i wrote one of them,
    i would be very proud. i hope the people who wrote those songs are happy. i hope
    they feel it's enough. i really do because they've made me happy. and i'm only
    one person.”
- source_sentence: “i can't eat and i can't sleep. i'm not doing well in terms of
    being a functional human, you know?”
  sentences:
  - “i have hated words and i have loved them, and i hope i have made them right.”
  - “i can't eat and i can't sleep. i'm not doing well in terms of being a functional
    human, you know?”
  - “the grand essentials to happiness in this life are something to do, something
    to love, and something to hope for.”
- source_sentence: “you must write every single day of your life... you must lurk
    in libraries and climb the stacks like ladders to sniff books like perfumes and
    wear books like hats upon your crazy heads... may you be in love every day for
    the next 20,000 days. and out of that love, remake a world.”
  sentences:
  - “you must write every single day of your life... you must lurk in libraries and
    climb the stacks like ladders to sniff books like perfumes and wear books like
    hats upon your crazy heads... may you be in love every day for the next 20,000
    days. and out of that love, remake a world.”
  - “there are no good girls gone wrong - just bad girls found out.”
  - “when i am with you, we stay up all night.when you're not here, i can't go to
    sleep.praise god for those two insomnias!and the difference between them.”
- source_sentence: “and when her lips met mine, i knew that i could live to be a hundred
    and visit every country in the world, but nothing would ever compare to that single
    moment when i first kissed the girl of my dreams and knew that my love would last
    forever.”
  sentences:
  - “the way to get started is to quit talking and begin doing. ”
  - “we fell in love, despite our differences, and once we did, something rare and
    beautiful was created. for me, love like that has only happened once, and that's
    why every minute we spent together has been seared in my memory. i'll never forget
    a single moment of it.”
  - “and when her lips met mine, i knew that i could live to be a hundred and visit
    every country in the world, but nothing would ever compare to that single moment
    when i first kissed the girl of my dreams and knew that my love would last forever.”
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '“and when her lips met mine, i knew that i could live to be a hundred and visit every country in the world, but nothing would ever compare to that single moment when i first kissed the girl of my dreams and knew that my love would last forever.”',
    '“and when her lips met mine, i knew that i could live to be a hundred and visit every country in the world, but nothing would ever compare to that single moment when i first kissed the girl of my dreams and knew that my love would last forever.”',
    "“we fell in love, despite our differences, and once we did, something rare and beautiful was created. for me, love like that has only happened once, and that's why every minute we spent together has been seared in my memory. i'll never forget a single moment of it.”",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 9 tokens</li><li>mean: 41.53 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 41.53 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>“freedom (n.): to ask nothing. to expect nothing. to depend on nothing.”</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <code>“freedom (n.): to ask nothing. to expect nothing. to depend on nothing.”</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | <code>1.0</code> |
  | <code>“i'm going to wake peeta," i say."no, wait," says finnick. "let's do it together. put our faces right in front of his."well, there's so little opportunity for fun left in my life, i agree. we position ourselves on either side of peeta, lean over until our faces are inches frim his nose, and give him a shake. "peeta. peeta, wake up," i say in a soft, singsong voice.his eyelids flutter open and then he jumps like we've stabbed him. "aa!"finnick and i fall back in the sand, laughing our heads off. every time we try to stop, we look at peeta's attempt to maintain a disdainful expression and it sets us off again.”</code> | <code>“i'm going to wake peeta," i say."no, wait," says finnick. "let's do it together. put our faces right in front of his."well, there's so little opportunity for fun left in my life, i agree. we position ourselves on either side of peeta, lean over until our faces are inches frim his nose, and give him a shake. "peeta. peeta, wake up," i say in a soft, singsong voice.his eyelids flutter open and then he jumps like we've stabbed him. "aa!"finnick and i fall back in the sand, laughing our heads off. every time we try to stop, we look at peeta's attempt to maintain a disdainful expression and it sets us off again.”</code> | <code>1.0</code> |
  | <code>“nothing of me is original. i am the combined effort of everyone i've ever known.”</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | <code>“nothing of me is original. i am the combined effort of everyone i've ever known.”</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 4.1.0
- Transformers: 4.52.3
- PyTorch: 2.6.0+cpu
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->