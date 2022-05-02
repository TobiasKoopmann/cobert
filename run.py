import uuid
import optuna
import argparse

from pathlib import Path
from util.factory import *
from util.training import *
from torch.optim.lr_scheduler import LambdaLR

import wandb

parser = argparse.ArgumentParser(description="Run BERT4Rec for Ranking/Prediction")
parser.add_argument("--model", type=str, default=None, help="String of the used model. ")
parser.add_argument("--dataset", type=str, default=None, help="String of the used dataset. ")
parser.add_argument("-d", "--data_dir", type=str, required=True, help="directory that contains all files for training")
parser.add_argument("-l", "--log_file", type=str, default=None, help="log results to this file if specified")
parser.add_argument("--save_predictions_file", type=str, default=None, help="save predictions to this file if specified")
parser.add_argument("--task", type=Bert4RecTask, choices=list(Bert4RecTask), default="ranking", help="task for training")
parser.add_argument("--trials", type=int, default=20, help="Trials for optuna optimisation")
parser.add_argument("--db_path", type=str, help="Path to optuna db")
parser.add_argument("--study_name", type=str, help="Name of optuna study")
parser.add_argument("-og", "--og_model", action="store_true", help="use the original bert4rec architecture (additional layer before output + weight tying)")
parser.add_argument("-seq", "--seq_model", action="store_true", help="use the sequential architecture (sequence consists of both authors and papers)")
parser.add_argument("-nova", "--nova_model", action="store_true", help="use the NOVA architecture")
parser.add_argument("-cobert", "--cobert_model", action="store_true", help="use the CoBERT architecture")
parser.add_argument("-w", "--weighted_embedding", action="store_true", help="use a weighted embedding")
parser.add_argument("-pe", "--paper_embedding", action="store_true", help="use an untrained paper embedding")
parser.add_argument("-ppe", "--pretrained_paper_embedding", action="store_true", help="use a pretrained paper embedding")
parser.add_argument("-pae", "--pretrained_author_embedding", action="store_true", help="use a pretrained author embedding, otherwise create a new one")
parser.add_argument("--no_train_embedding", action="store_true", help="if flagged, the embeddings will be static and not be further trained")
parser.add_argument("--ks", type=int, nargs="+", default=(1, 5, 10), help="every k used for calculating NDCG/Hit @ k, e. g. '--ks 1 2 3'")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("--validate_every_n", type=int, default=3, help="validate every n-th epoch during training")
parser.add_argument("--n_early_stop", type=int, default=5, help="if specified, early stop training if validation loss was not improved after this many validation epochs")
parser.add_argument("-bs", "--batch_size", type=int, default=64)
parser.add_argument("-ebs", "--effective_batch_size", type=int, default=64)
parser.add_argument("-dim", "--hidden_size", type=int, default=768, help="hidden dim of the transformer layers (if using pretrained embeddings make sure dims match)")
parser.add_argument("--n_layers", type=int, default=2, help="amount of transformer layers")
parser.add_argument("--n_heads", type=int, default=8, help="amount of heads in multihead self-attention")
parser.add_argument("--max_len", type=int, default=30, help="max sequence length of co-authors, longer sequence will be truncated from older to newer")
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--clip_grad_norm", type=float, default=5., help="clip gradients whose norm exceeds this value")
parser.add_argument("--dropout", type=float, default=1e-1)
parser.add_argument("--p_mlm", type=float, default=2e-1, help="probability of masking tokens during training (MLM)")
parser.add_argument("--p_force_last", type=float, default=5e-2, help="probability of masking only the last token during training (MLM)")
parser.add_argument("--p_mask_max", type=float, default=5e-1, help="maximum percentage of the sequence that will be masked during training (MLM)")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="run model on cpu/cuda, defaults to cpu if cuda is unavailable")
parser.add_argument("--workers", type=int, default=1, help="How many workers for the dataloader")
parser.add_argument("--use_negative_sampling", type=bool, default=False, help="Name says it all")
parser.add_argument("--verbose", type=int, default=1, help="How loud it is")


def objective(trial):
    args = vars(parser.parse_args())
    curr_id = str(uuid.uuid4())
    args["id"] = curr_id
    args["run_dir"] = f"data/journal_runs/{curr_id}/"
    args["save_predictions_file"] = f"data/journal_runs/{curr_id}/predictions"
    args["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 0.005)
    args["hidden_size"] = 768
    args["n_layers"] = trial.suggest_int("n_layers", 2, 2)
    args["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 6, 8])
    args["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.05)
    args["dropout"] = trial.suggest_float("dropout", 1e-2, 0.5)
    args["p_mlm"] = trial.suggest_float("p_mlm", 1e-2, 0.5)
    args["p_force_last"] = trial.suggest_float("p_force_last", 1e-2, 0.5)
    args["p_mask_max"] = trial.suggest_float("p_mask_max", 1e-2, 0.5)
    args["trial_id"] = trial.number

    if args["seq_model"]:
        args['max_len'] = 15

    if args["verbose"]:
        print("Config:")
        for k, v in args.items():
            print(k, ":", v)

    log_handlers = [logging.StreamHandler()]
    log_file = os.path.join(args["run_dir"], "run.logs")
    parent_dir = os.path.dirname(log_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    log_handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=log_handlers
    )

    if not os.path.exists(args["run_dir"]):
        os.makedirs(args["run_dir"])
    json.dump(args, open(os.path.join(args["run_dir"], "config.json"), 'w', encoding='utf-8'))
    print("Test Args are:")
    for k, v in args.items():
        print(f"\t{k} \t- {v}")
    return main(args=args)


def main(args: dict) -> float:
    # used for ignoring things like loss for non masked labels, evaluation, serialization ...
    ignore_index = -100

    device = torch.device(args["device"])

    dataloader_train, dataloader_val, dataloader_test = \
        get_data_loaders(data_dir=args["data_dir"],
                         task=args["task"],
                         sequential=args["seq_model"] if "seq_model" in args else None,
                         bucket=args["bucket_embedding"] if "bucket_embedding" in args else None,
                         batch_size=args["batch_size"],
                         max_len=args["max_len"],
                         p_mlm=args["p_mlm"],
                         p_mask_max=args["p_mask_max"],
                         ignore_index=ignore_index,
                         num_workers=args["workers"] if "workers" in args else 1)
    model = get_model(args).to(device)    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
    if "wandb" not in args or args['wandb']:
        wandb.init(project=f"cobert", entity="lsx", reinit=True)
        wandb.run.name = f"cobert-{args['model']}-{args['dataset']}-{args['trial_id']}"
        wandb.run.save()
        wandb.config = args

    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = LambdaLR(optimizer, lambda epoch: 0.1)
    evaluation = Evaluation(n_samples_file=Path(args['data_dir'], "negative-samples.json"), ks=args["ks"], ignore_index=ignore_index)
    callbacks = [evaluation]

    # min_loss = float("inf")
    max_k = max(args["ks"])
    max_ndcg_k = -float("inf")
    n_tries = args["epochs"] if "n_early_stop" not in args else args["n_early_stop"]
    final_epoch = 0
    if "wandb" not in args or args['wandb']:
        wandb.log({"Learning rate": scheduler.get_last_lr()[0]})
    for i in range(1, args["epochs"] + 1):
        loss = train_one_epoch(model=model, dataloader=dataloader_train, criterion=criterion, optimizer=optimizer,
                               device=device, batch_size=args["batch_size"], effective_bs=args["effective_batch_size"],
                               clip_grad_norm=args["clip_grad_norm"], epoch=i)
        scheduler.step()

        if "wandb" not in args or args['wandb']:
            wandb.log({"Loss/train": loss})
            wandb.log({"Learning rate": scheduler.get_last_lr()[0]})

        logging.info(f"epoch {i} | train | loss={loss}")

        if i % args["validate_every_n"] != 0:
            continue

        loss = evaluate(model, dataloader_val, criterion, device, callbacks, i)
        if "wandb" not in args or args['wandb']:
            wandb.log({"Loss/val": loss})
        logging.info(f"epoch {i} | validate | loss={loss} {str(evaluation)}")
        ndcg_k = evaluation.evaluation["sampled_ndcg"][max_k]
        val_ndcg = evaluation.get_metric('ndcg')
        val_hit = evaluation.get_metric('hit')
        val_sampled_ndcg = evaluation.get_metric('sampled_ndcg')
        val_sampled_hit = evaluation.get_metric('sampled_hit')
        if "wandb" not in args or args['wandb']:
            for curr_k, score in val_ndcg:
                wandb.log({f"NDCG/val@{curr_k}": score})
            for curr_k, score in val_hit:
                wandb.log({f"HIT/val@{curr_k}": score})
            for curr_k, score in val_sampled_ndcg:
                wandb.log({f"SampledNDCG/val@{curr_k}": score})
            for curr_k, score in val_sampled_hit:
                wandb.log({f"SampledHIT/val@{curr_k}": score})

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

    if "save_predictions_file" in args:
        evaluation.reset()
        serializer = get_serializer(ignore_index, args["save_predictions_file"])
        callbacks.append(serializer)
        loss = evaluate(model, dataloader_test, criterion, device, callbacks, test=True)
        logging.info(f"test | loss={loss} {str(evaluation)}")
    else:
        loss = evaluate(model, dataloader_test, criterion, device, callbacks, test=True)
        logging.info(f"test | loss={loss} {str(evaluation)}")
    args['loss'] = loss
    args['hit'] = evaluation.get_metric('hit')
    args['ndcg'] = evaluation.get_metric('ndcg')
    args['sampled_hit'] = evaluation.get_metric('sampled_hit')
    args['sampled_ndcg'] = evaluation.get_metric('sampled_ndcg')
    args['val_hit'] = val_hit
    args['val_ndcg'] = val_ndcg
    args['val_sampled_hit'] = val_sampled_hit
    args['val_sampled_ndcg'] = val_sampled_ndcg
    args['epoch'] = final_epoch
    with open(os.path.join(os.path.dirname(args["run_dir"]), "..", "results.json"), 'a', encoding='utf-8') as f:
        json.dump(args, f)
        f.write("\n")
    if "save_model" in args:
        torch.save(model.state_dict(), os.path.join(args["run_dir"], "model.pt"))
    return max_ndcg_k


if __name__ == '__main__':
    args = vars(parser.parse_args())
    study = optuna.create_study(direction='maximize', study_name=args['study_name'],
                                storage=f'sqlite:///{args["db_path"]}', load_if_exists=True)
    study.optimize(objective, n_trials=args['trials'])
