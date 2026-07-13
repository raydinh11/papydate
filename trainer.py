import math
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

log = logging.getLogger("trainer")


def eps_step(data_loader, model, cfg_train, optimizer=None, scheduler=None,
             mode="train", scaler=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    loss_name = cfg_train["loss_name"]
    losses = [0 for _ in loss_name]
    losses_count = [0 for _ in loss_name]

    len_data_loader = len(data_loader)
    iter_cap = int(cfg_train.get("iters_per_epoch", 0)) if mode == "train" else 0
    total_iters = min(len_data_loader, iter_cap) if iter_cap > 0 else len_data_loader

    pbar = tqdm(enumerate(data_loader), total=total_iters, leave=False, desc=mode)

    loss_comp = []
    for i, batch in pbar:
        if iter_cap > 0 and i >= iter_cap:
            break
        if mode == "train" and optimizer is not None:
            optimizer.zero_grad()
            opt_stepped = True
            if scaler is not None and scaler.is_enabled():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss, loss_comp = model.forward_train(batch, cfg_train)
                if torch.isnan(torch.sum(loss)):
                    log.error("LOSS Nan!"); break
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                opt_stepped = scaler.get_scale() >= scale_before
            else:
                loss, loss_comp = model.forward_train(batch, cfg_train)
                if torch.isnan(torch.sum(loss)):
                    log.error("LOSS Nan!"); break
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                optimizer.step()
            if scheduler is not None and opt_stepped:
                scheduler.step()
        else:
            amp_eval = scaler is not None and scaler.is_enabled()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_eval):
                loss, loss_comp = model.forward_train(batch, cfg_train)

        with torch.no_grad():
            len_info = min(len(loss_name), len(loss_comp))
            for idx, loss_tmp in enumerate(loss_comp[:len_info]):
                losses[idx] += loss_tmp.sum().item()
                losses_count[idx] += 1
            pbar.set_postfix({name: losses[idx] / max(losses_count[idx], 1)
                              for idx, name in enumerate(loss_name[:len_info])}, refresh=False)

    len_info = min(len(loss_name), len(loss_comp))
    loss_info = {loss_name[idx]: val / max(losses_count[idx], 1)
                 for idx, val in enumerate(losses[:len_info])}

    return loss_info


def setup_opt(model, cfg_train):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "LayerNorm.weight" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW([
        {"params": decay, "weight_decay": float(cfg_train["weight_decay"])},
        {"params": no_decay, "weight_decay": 0},
    ], lr=float(cfg_train["lr"]))


def training_loop(model, train_loader, valid_loader, cfg_train):
    model = model.to(cfg_train["device"])
    n_eps = cfg_train["n_eps"]

    optimizer = setup_opt(model, cfg_train)
    amp_enabled = bool(cfg_train.get("amp", True)) and cfg_train["device"].startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    iter_cap = int(cfg_train.get("iters_per_epoch", 0))
    full_steps = len(train_loader)
    steps_per_epoch = min(full_steps, iter_cap) if iter_cap > 0 else full_steps
    total_steps = n_eps * steps_per_epoch

    def lr_lambda(step):
        progress = float(step) / float(max(1, total_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    def cond(epoch):
        return (epoch % cfg_train["eval_step"] == 0) or epoch == n_eps - 1

    for epoch in range(n_eps):
        msg = f"Epoch {epoch}/{n_eps} "
        loss_train = eps_step(train_loader, model, cfg_train, optimizer, scheduler, mode="train", scaler=scaler)
        log.info("[TRAIN] " + msg + " ".join("%s: %.3f " % (k, v) for k, v in loss_train.items()))

        if cond(epoch):
            loss_valid = eps_step(valid_loader, model, cfg_train, None, None, mode="valid", scaler=scaler)
            log.info("[VALID] " + msg + " ".join("%s: %.3f " % (k, v) for k, v in loss_valid.items()))

            torch.save(model.state_dict(), "%s/end_eps.pt" % cfg_train["ckpt"])

            if cfg_train.get("save_every_eval", False):
                torch.save(model.state_dict(), "%s/eps_%03d.pt" % (cfg_train["ckpt"], epoch))