# CoBERT: Scientific Collaboration Prediction viaSequential Recommendation

Recently a lot of work based on the transformer and BERT architecture has been published. One of them is the work by Sun et al., which adopts the BERT architecture on recommendation. The use sequences of IDs from items to train the BERT architecture end-to-end.

Our idea is to combine two BERT architectures. Firstly we want to create a latent representation for each author. The approach would be to apply a language model on all texts from an author. This creates a vector representing all the scientific work of an author.

These representation then can be used as input for out own BERT4Rec. As sequences of clicked items, we generate random (or weighted random) walks based on the co-author graph. The actual recommendation task here would be to recommend a suitable cooperation for an author in the graph.

# Abstract 
Collaborations are an important factor for scientific success,  as  the  joint  work  leads  to  results  individual  scientists cannot easily reach. 
Recommending collaborations automatically can alleviate the time-consuming and tedious search for potential collaborators. 
Usually,  such  recommendation  systems  rely on graph structures modeling co-authorship of papers and content-based relations such as similar paper keywords. 
Models are then trained to estimate the probability of links between certain authors in these  graphs. 
In this paper, we argue that the order of papers is crucial for reliably predicting future collaborations, which is not considered by graph-based recommendation systems. 
We thus propose to reformulate the task of collaboration recommendation as a sequential recommendation task. 
Here, we aim to predict the next co-author in a chronologically sorted sequence of an author’s collaborators. 
We introduce CoBERT, a BERT4Rec inspired model, that predicts the sequence’s next co-author and thus a potential collaborator. 
Since the order of co-authors of a single paper is not that important compared to the overall paper order,we leverage positional embeddings encoding paper positions instead of co-author positions in the sequence. 
Additionally, we inject content features about every paper and their co-authors. 
We evaluate CoBERT on two datasets consisting of papers from the field of Artificial Intelligence and the journal PlosOne. 
We show that CoBERT can outperform graph-based methods and BERT4Rec when predicting the co-authors of the next paper. 
We make our code and data available.

# Installation

Pip (install cuda yourself):

```
pip install -r requirements.txt
```

# Usage

There is a [command line interface](/run.py) intended for training and evaluation.

## OG Model

- Specify `--data_dir`
- To save logs and predictions add `--log_file` and `--save_predictions_dir`
- Select between ranking and predict for `--task`
- Choose `--hidden_size` between 64 / 128 / 256
- All other parameters below match the original implementation

```
python run.py  \
--data_dir util/files-n10 \
--log_file ./run.log \
--save_predictions_file ./predictions.json \
--task ranking \
--og_model \
--learning_rate 1e-3 \
--validate_every_n 3 \
--n_early_stop 1 \
--epochs 10000 \
--n_early_stop 3 \
--batch_size 256 \
--hidden_size 64 \
--n_heads 2 \
--n_layers 2 \
--max_len 50 \
--dropout 0.1
```

## Reformed Implementation

- Specify `--data_dir`
- To save logs and predictions add `--log_file` and `--save_predictions_dir`
- Select between ranking and predict for `--task`
- `--pretrained_author_embedding` loads the pretrained author embedding
- `--paper_embedding` enables an untrained paper embedding
- `--pretrained_paper_embedding` loads the pretrained paper embedding
- If `--no-train-embeddings` is specified, the pretrained embeddings will not be trained any further

```
python run.py  \
--data_dir util/files-n10 \
--log_file ./run.log \
--save_predictions_file ./predictions.json \
--pretrained_author_embedding \
--pretrained_paper_embedding \
--task ranking \
--learning_rate 1e-3 \
--epochs 10000 \
--validate_every_n 3 \
--n_early_stop 1 \
--batch_size 200 \
--hidden_size 768 \
--n_heads 8 \
--n_layers 2 \
--max_len 50 \
--dropout 1e-1
```

## Weighted Embeddings

- Specify `--weighted_embedding`
- Works with all combinations of `--pretrained_author_embedding`, `--paper_embedding`, `--pretrained_paper_embedding`

## Sequential Model

- Specify `--seq_model`

## Potential Hyperparameters to be tuned

- `--learning_rate`
- `--p_mlm`: Percentage of masked tokens during training
- `--p_force_last`: Percentage of sequences where only the last token will be masked during training (to match the test task)
- `--n_layers`
- `--n_heads`
- `--dropout`
- `--weight_decay`
- `--max_len`: Maximum co-author sequence length (longer sequences will be truncated starting from oldest to newest)


## Complete API

There are several more possible arguments:

```
usage: run.py [-h] -d DATA_DIR [-l LOG_FILE] [--save_predictions_file SAVE_PREDICTIONS_FILE] [--task {ranking,predict}] [-og] [-seq] [-w] [-pe] [-ppe]
              [-pae] [--no_train_embedding] [--ks KS [KS ...]] [-lr LEARNING_RATE] [-e EPOCHS] [--validate_every_n VALIDATE_EVERY_N]
              [--n_early_stop N_EARLY_STOP] [-bs BATCH_SIZE] [-dim HIDDEN_SIZE] [--n_layers N_LAYERS] [--n_heads N_HEADS] [--max_len MAX_LEN]
              [--weight_decay WEIGHT_DECAY] [--clip_grad_norm CLIP_GRAD_NORM] [--dropout DROPOUT] [--p_mlm P_MLM] [--p_force_last P_FORCE_LAST]
              [--p_mask_max P_MASK_MAX] [--device {cpu,cuda}]

Run BERT4Rec for Ranking/Prediction

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        directory that contains all files for training
  -l LOG_FILE, --log_file LOG_FILE
                        log results to this file if specified
  --save_predictions_file SAVE_PREDICTIONS_FILE
                        save predictions to this file if specified
  --task {ranking,predict}
                        task for training
  -og, --og_model       use the original bert4rec architecture (additional layer before output + weight tying)
  -seq, --seq_model     use the sequential architecture (sequence consists of both authors and papers)
  -w, --weighted_embedding
                        use a weighted embedding
  -pe, --paper_embedding
                        use an untrained paper embedding
  -ppe, --pretrained_paper_embedding
                        use a pretrained paper embedding
  -pae, --pretrained_author_embedding
                        use a pretrained author embedding, otherwise create a new one
  --no_train_embedding  if flagged, the embeddings will be static and not be further trained
  --ks KS [KS ...]      every k used for calculating NDCG/Hit @ k, e. g. '--ks 1 2 3'
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -e EPOCHS, --epochs EPOCHS
  --validate_every_n VALIDATE_EVERY_N
                        validate every n-th epoch during training
  --n_early_stop N_EARLY_STOP
                        if specified, early stop training if validation loss was not improved after this many validation epochs
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -dim HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        hidden dim of the transformer layers (if using pretrained embeddings make sure dims match)
  --n_layers N_LAYERS   amount of transformer layers
  --n_heads N_HEADS     amount of heads in multihead self-attention
  --max_len MAX_LEN     max sequence length of co-authors, longer sequence will be truncated from older to newer
  --weight_decay WEIGHT_DECAY
  --clip_grad_norm CLIP_GRAD_NORM
                        clip gradients whose norm exceeds this value
  --dropout DROPOUT
  --p_mlm P_MLM         probability of masking tokens during training (MLM)
  --p_force_last P_FORCE_LAST
                        probability of masking only the last token during training (MLM)
  --p_mask_max P_MASK_MAX
                        maximum percentage of the sequence that will be masked during training (MLM)
  --device {cpu,cuda}   run model on cpu/cuda, defaults to cpu if cuda is unavailable
```
