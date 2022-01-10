import optuna

from util.factory import *
from util.training import *


def objective(trial):
    args = json.load(open("tmp.json", 'r'))
    args["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 0.5)
    args["n_early_stop"] = trial.suggest_int("n_early_stop", 1, 10)
    args["hidden_size"] = 768
    args["n_layers"] = trial.suggest_int("n_layers", 1, 8)
    args["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 6, 8, 12, 16])
    args["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.5)
    args["dropout"] = trial.suggest_float("dropout", 1e-2, 0.5)
    args["p_mlm"] = trial.suggest_float("p_mlm", 1e-2, 0.5)
    args["p_force_last"] = trial.suggest_float("p_force_last", 1e-2, 0.5)
    args["p_mask_max"] = trial.suggest_float("p_mask_max", 1e-2, 0.5)

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
    if args["verbose"]:
        print("Config:")
        for k, v in args.items():
            print(k, ":", v)
    return main(args=args)


def main(args: dict) -> float:
    # used for ignoring things like loss for non masked labels, evaluation, serialization ...
    ignore_index = -100

    device = torch.device(args["device"])

    dataloader_train, dataloader_val, dataloader_test = get_data_loaders(data_dir=args["data_dir"],
                                                                         task=args["task"],
                                                                         sequential=args["seq-model"] if "seq-model" in args else None,
                                                                         bucket=args["bucket_embedding"] if "bucket_embedding" in args else None,
                                                                         batch_size=args["batch_size"],
                                                                         max_len=args["max_len"],
                                                                         p_mlm=args["p_mlm"],
                                                                         p_mask_max=args["p_mask_max"],
                                                                         ignore_index=ignore_index,
                                                                         num_workers=args["workers"] if "workers" in args else 1)
    model = get_model(args).to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)

    if args["weight_decay"] > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    evaluation = get_evaluator(args["data_dir"], args["use_negative_sampling"], ks=args["ks"], ignore_index=ignore_index)
    callbacks = [evaluation]

    # min_loss = float("inf")
    min_k = min(args["ks"])
    max_ndcg_k = -float("inf")
    n_tries = args["epochs"] if "n_early_stop" not in args else args["n_early_stop"]
    final_epoch = 0
    for i in range(1, args["epochs"] + 1):
        loss = train_one_epoch(model=model, dataloader=dataloader_train, criterion=criterion, optimizer=optimizer,
                               device=device, batch_size=args["batch_size"], effective_bs=args["effective_batch_size"],
                               clip_grad_norm=args["clip_grad_norm"], epoch=i)
        logging.info(f"epoch {i} | train | loss={loss}")

        if i % args["validate_every_n"] != 0:
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
    args['val_hit'] = val_hit
    args['val_ndcg'] = val_ndcg
    args['epoch'] = final_epoch
    with open(os.path.join(os.path.dirname(args["run_dir"]), "..", "results.json"), 'a', encoding='utf-8') as f:
        json.dump(args, f)
        f.write("\n")
    if "save_model" in args:
        torch.save(model.state_dict(), os.path.join(args["run_dir"], "model.pt"))
    return loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
