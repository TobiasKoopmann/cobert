from typing import Optional, Callable, Iterable

import torch
import tqdm


def train_one_epoch(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    batch_size: int,
                    effective_bs: int,
                    clip_grad_norm: float = 5.,
                    epoch: Optional[int] = None):
    model = model.train()
    pbar = tqdm.tqdm(enumerate(dataloader, start=1), total=len(dataloader))
    total_loss = 0

    backprop_every = int(effective_bs / batch_size)
    for i, batch in pbar:
        for key in batch:
            try:
                batch[key] = batch[key].to(device)
            except:
                continue
        logits = model(batch)

        # logits = model(batch["author_ids"], batch["attention_masks"], batch["position_ids"], batch["paper_ids"])

        loss = criterion(logits.transpose(2, 1), batch["labels"])
        if i % backprop_every == 0:
            model.zero_grad()
            loss.backward()

            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            total_loss -= (total_loss / i) - (loss.item() / i)

            pbar_description = f"train | batch {i:d}/{len(dataloader)} | loss {total_loss:.2f}"
            if epoch:
                pbar_description = f"epoch {epoch} | " + pbar_description
            pbar.set_description(pbar_description)

    return total_loss


def evaluate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             device: torch.device,
             callbacks: Iterable[Callable] = (),
             epoch: Optional[int] = None,
             test: bool = False):
    model = model.eval()
    pbar = tqdm.tqdm(enumerate(dataloader, start=1), total=len(dataloader))

    total_loss = 0
    with torch.no_grad():
        for i, batch in pbar:
            for key in batch:
                try:
                    batch[key] = batch[key].to(device)
                except:
                    continue

            logits = model(batch)
            # logits = model(batch["author_ids"], batch["attention_masks"], batch["position_ids"], batch["paper_ids"])

            loss = criterion(logits.transpose(2, 1), batch["labels"])

            total_loss -= (total_loss / i) - (loss.item() / i)

            for callback in callbacks:
                callback(logits, batch["labels"])

            pbar_description = f"{'test' if test else 'validate'} | batch {i:d}/{len(dataloader)} | loss {total_loss:.2f}"
            if epoch:
                pbar_description = f"epoch {epoch} | " + pbar_description
            pbar.set_description(pbar_description)

    return total_loss