import argparse
import logging

from util.factory import *
from util.training import *

parser = argparse.ArgumentParser(description="Run BERT4Rec for Ranking/Prediction")
parser.add_argument("-d", "--data_dir",
                    type=str,
                    required=True,
                    help="directory that contains all files for training")
parser.add_argument("--id",
                    type=str,
                    required=True,
                    help="the id of the run")
parser.add_argument("--run_dir",
                    type=str,
                    required=True,
                    help="directory of the run")
parser.add_argument("-l", "--log_file",
                    type=str,
                    default=None,
                    help="log results to this file if specified")
parser.add_argument("--save_predictions_file",
                    type=str,
                    default=None,
                    help="save predictions to this file if specified")
parser.add_argument("--task",
                    type=Bert4RecTask,
                    choices=list(Bert4RecTask),
                    default="ranking",
                    help="task for training")
parser.add_argument("-og", "--og_model",
                    action="store_true",
                    help="use the original bert4rec architecture (additional layer before output + weight tying)")
parser.add_argument("-be", "--bucket_embedding",
                    action="store_true",
                    help="use the original bert4rec architecture and bucket embeddings (additional layer before output + weight tying)")
parser.add_argument("-seq", "--seq_model",
                    action="store_true",
                    help="use the sequential architecture (sequence consists of both authors and papers)")
parser.add_argument("-w", "--weighted_embedding",
                    action="store_true",
                    help="use a weighted embedding")
parser.add_argument("-pe", "--paper_embedding",
                    action="store_true",
                    help="use an untrained paper embedding")
parser.add_argument("-ppe", "--pretrained_paper_embedding",
                    action="store_true",
                    help="use a pretrained paper embedding")
parser.add_argument("-pae", "--pretrained_author_embedding",
                    action="store_true",
                    help="use a pretrained author embedding, otherwise create a new one")
parser.add_argument("--no_train_embedding",
                    action="store_true",
                    help="if flagged, the embeddings will be static and not be further trained")
parser.add_argument("--ks",
                    type=int,  # Iterable[int]
                    nargs="+",
                    default=(1, 5, 10),
                    help="every k used for calculating NDCG/Hit @ k, e. g. '--ks 1 2 3'")
parser.add_argument("-lr", "--learning_rate",
                    type=float,
                    default=1e-3)
parser.add_argument("-e", "--epochs",
                    type=int,
                    default=100)
parser.add_argument("--validate_every_n",
                    type=int,
                    default=2,
                    help="validate every n-th epoch during training")
parser.add_argument("--n_early_stop",
                    type=int,  # Optional[int]
                    default=5,
                    help="if specified, early stop training if validation loss was not improved after this many validation epochs")
parser.add_argument("-bs", "--batch_size",
                    type=int,
                    default=64)
parser.add_argument("--effective_batch_size",
                    type=int,
                    default=64)
parser.add_argument("-dim", "--hidden_size",
                    type=int,
                    default=768,
                    help="hidden dim of the transformer layers (if using pretrained embeddings make sure dims match)")
parser.add_argument("--n_layers",
                    type=int,
                    default=2,
                    help="amount of transformer layers")
parser.add_argument("--n_heads",
                    type=int,
                    default=8,
                    help="amount of heads in multi-head self-attention")
parser.add_argument("--max_len",
                    type=int,
                    default=50,
                    help="max sequence length of co-authors, longer sequence will be truncated from older to newer")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.)
parser.add_argument("--clip_grad_norm",
                    type=float,
                    default=5.,
                    help="clip gradients whose norm exceeds this value")
parser.add_argument("--dropout",
                    type=float,
                    default=1e-1)
parser.add_argument("--p_mlm",
                    type=float,
                    default=2e-1,
                    help="probability of masking tokens during training (MLM)")
parser.add_argument("--p_force_last",
                    type=float,
                    default=5e-2,
                    help="probability of masking only the last token during training (MLM)")
parser.add_argument("--p_mask_max",
                    type=float,
                    default=5e-1,
                    help="maximum percentage of the sequence that will be masked during training (MLM)")
parser.add_argument("--device",
                    type=str,
                    choices=["cpu", "cuda"],
                    default="cuda",
                    help="run model on cpu/cuda, defaults to cpu if cuda is unavailable")
parser.add_argument("--workers",
                    type=int,
                    default=1,
                    help="How many workers are used for the dataloader")
parser.add_argument("--validation_run",
                    type=str,
                    default=None,
                    help="Id of the original run config")
parser.add_argument("--no-paramsearch",
                    action="store_true",
                    help="If yes, searching for hyper params. ")

args = parser.parse_args()


def main(args):
    # used for ignoring things like loss for non masked labels, evaluation, serialization ...
    ignore_index = -100

    device = torch.device(args.device)

    dataloader_train, dataloader_val, dataloader_test = get_data_loaders(data_dir=args.data_dir,
                                                                         task=args.task,
                                                                         sequential=args.seq_model,
                                                                         bucket=args.bucket_embedding,
                                                                         batch_size=args.batch_size,
                                                                         max_len=args.max_len,
                                                                         p_mlm=args.p_mlm,
                                                                         p_mask_max=args.p_mask_max,
                                                                         ignore_index=ignore_index,
                                                                         num_workers=args.workers)
    model = get_model(args).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)

    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    evaluation = get_evaluator(args.data_dir, ks=args.ks, ignore_index=ignore_index)
    callbacks = [evaluation]

    # min_loss = float("inf")
    min_k = min(args.ks)
    max_ndcg_k = -float("inf")
    n_tries = args.epochs if args.n_early_stop is None else args.n_early_stop
    final_epoch = 0
    for i in range(1, args.epochs + 1):
        loss = train_one_epoch(model=model, dataloader=dataloader_train, criterion=criterion, optimizer=optimizer,
                               device=device, batch_size=args.batch_size, effective_bs=args.effective_batch_size,
                               clip_grad_norm=args.clip_grad_norm, epoch=i)
        logging.info(f"epoch {i} | train | loss={loss}")

        if i % args.validate_every_n != 0:
            continue

        loss = evaluate(model, dataloader_val, criterion, device, callbacks, i)
        logging.info(f"epoch {i} | validate | loss={loss} {str(evaluation)}")
        ndcg_k = evaluation.evaluation["ndcg"][min_k]
        val_ndcg = evaluation.get_metric('ndcg')
        val_hit = evaluation.get_metric('hit')
        evaluation.reset()

        # early stopping
        if ndcg_k <= max_ndcg_k and n_tries > 0:
            n_tries -= 1
            continue
        elif ndcg_k <= max_ndcg_k:
            logging.info(f"early stopped at epoch {i}")
            final_epoch = i
            break

        max_ndcg_k = ndcg_k

    if args.save_predictions_file:
        evaluation.reset()
        serializer = get_serializer(ignore_index, args.save_predictions_file)
        callbacks.append(serializer)
        loss = evaluate(model, dataloader_test, criterion, device, callbacks, test=True)
        logging.info(f"test | loss={loss} {str(evaluation)}")
    else:
        loss = evaluate(model, dataloader_test, criterion, device, callbacks, test=True)
        logging.info(f"test | loss={loss} {str(evaluation)}")
    args_dict = vars(args)
    args_dict['loss'] = loss
    args_dict['hit'] = evaluation.get_metric('hit')
    args_dict['ndcg'] = evaluation.get_metric('ndcg')
    args_dict['val_hit'] = val_hit
    args_dict['val_ndcg'] = val_ndcg
    args_dict['epoch'] = final_epoch
    with open(os.path.join(os.path.dirname(args.run_dir), "..", "results.json"), 'a', encoding='utf-8') as f:
        json.dump(args_dict, f)
        f.write("\n")
    print("Finshed. ")
    torch.save(model.state_dict(), os.path.join(args.run_dir, "model.pt"))


if __name__ == '__main__':
    args = parser.parse_args()

    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.join(args.run_dir, "run.logs")
        parent_dir = os.path.dirname(log_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        log_handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=log_handlers
    )

    args_dict = vars(args)
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    json.dump(args_dict, open(os.path.join(args.run_dir, "config.json"), 'w', encoding='utf-8'))
    print("Config:")
    for k, v in args_dict.items():
        print(k, ":", v)
    main(args)
