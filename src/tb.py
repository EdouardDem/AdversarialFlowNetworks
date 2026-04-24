import os
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch import nn
from omegaconf import DictConfig

from src.envs.base_env import BaseEnv
from src.data import gen_batch_traj_buffer, TrajectoryBuffer
from src.eval import test_agent, UniformAgent

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def check_model(model: nn.Module) -> None:
    """Validate that all model parameters and buffers are finite.

    Raises ``ValueError`` if any NaN / ±Inf values are found, listing the offending
    tensors. Use after loading a checkpoint — a diverged run is not recoverable by
    zeroing weights, so fail loudly and let the caller fall back to an earlier ckpt.
    """
    bad = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        n_bad = int((~torch.isfinite(tensor)).sum().item())
        if n_bad:
            bad.append((name, n_bad, tensor.numel()))

    if bad:
        lines = [f"  {name}: {n}/{total} non-finite" for name, n, total in bad]
        raise ValueError(
            "Corrupted checkpoint — non-finite values in model weights:\n"
            + "\n".join(lines)
            + "\nLoad an earlier checkpoint instead."
        )


def compute_loss(afn: nn.Module, batch: tuple[torch.Tensor, ...]):
    states, masks, curr_players, actions, dones, log_reward = batch
    batch_size, traj_len, _, _, _ = states.shape

    log_numerator = torch.zeros(batch_size, device=DEVICE)
    log_denominator = torch.zeros(batch_size, device=DEVICE)

    log_numerator += afn.log_Z_0

    for t in range(traj_len):
        curr_states = states[:, t, :, :, :].squeeze(1)
        curr_masks = masks[:, t, :].squeeze(1)
        curr_player = t % 2
        curr_actions = actions[:, t].long()
        curr_not_dones = (dones[:, t].squeeze(1) == 1).bool()
        curr_terminal = (dones[:, t].squeeze(1) == 2).bool()

        if torch.any(curr_not_dones) or torch.any(curr_terminal):
            _, policy = afn(curr_states, curr_player)
            policy = policy * curr_masks - (1 - curr_masks) * 1e9
            probs = nn.functional.softmax(policy, dim=1)
            probs = probs.gather(1, curr_actions).squeeze(1)
            num_children = curr_masks.sum(dim=1)

            if curr_player == 0:
                log_numerator[curr_not_dones] += probs[curr_not_dones].log()
                log_numerator[curr_not_dones] += num_children[curr_not_dones].log()

            if curr_player == 1:
                log_denominator[curr_not_dones] += probs[curr_not_dones].log()
                log_denominator[curr_not_dones] += num_children[curr_not_dones].log()

            if torch.any(curr_terminal):
                log_denominator[curr_terminal] += log_reward[curr_terminal, 0]

    loss = F.mse_loss(log_numerator, log_denominator)

    return loss


def train_afn(
    afn: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: TrajectoryBuffer,
    batch_size=2048,
):
    batch = buffer.sample(batch_size)
    optimizer.zero_grad()

    loss = compute_loss(afn, batch)
    loss.backward()

    optimizer.step()

    return loss.item()


def train(
    env: BaseEnv,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: TrajectoryBuffer,
    cfg: DictConfig,
):
    resume_from = cfg.get("resume_from", None)
    start_step = 0

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # Legacy checkpoint: the full module was pickled under "model".
            model.load_state_dict(ckpt["model"].state_dict())

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("Legacy checkpoint: optimizer state not restored.")

        if "step" in ckpt:
            start_step = ckpt["step"] + 1
        else:
            # Recover from filename like ckpt-00015000.pt.
            stem = os.path.splitext(os.path.basename(resume_from))[0]
            start_step = int(stem.split("-")[-1]) + 1

        check_model(model)

        run_dir = os.path.dirname(os.path.abspath(resume_from))
    else:
        run_dir = os.path.join(
            cfg.ckpt_dir, wandb.run.id if wandb.run else wandb.util.generate_id()
        )
        os.makedirs(run_dir, exist_ok=True)

    print("Generating initial buffer")
    buffer = gen_batch_traj_buffer(
        buffer=buffer,
        env=env,
        afn=model,
        num_trajectories=cfg.num_initial_traj,
        batch_size=cfg.buffer_batch_size,
    )

    train_pbar = tqdm(
        range(start_step, cfg.total_steps),
        initial=start_step,
        total=cfg.total_steps,
        leave=False,
    )
    train_pbar.set_description("Train")
    for step in train_pbar:
        # Train
        loss = train_afn(
            afn=model, optimizer=optimizer, buffer=buffer, batch_size=cfg.batch_size
        )
        wandb.log({"loss": loss, "step": step}) if wandb.run else None

        # Eval
        if step and step % cfg.eval_every == 0:
            print("-" * 10, " Eval ", "-" * 10)
            model.eval()

            try:
                test_res = test_agent(env, model, UniformAgent())
                test_res.update({"step": step})
                print(test_res)
                wandb.log(test_res) if wandb.run else None
            except Exception as e:
                print(f"Error during evaluation: {e}")    

            model.train()

            # Save the checkpoint
            last_ckpt_path = os.path.join(run_dir, f"ckpt-{step:08}.pt")
            ckpt = {
                "env": env,
                "model": model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
            }
            torch.save(ckpt, last_ckpt_path)
            print(f"Saved checkpoint at {last_ckpt_path}")

        # Sample new trajectories
        if step and step % cfg.eval_every == 0:
            print("-" * 10, " Regen ", "-" * 10)
            model.eval()
            buffer = gen_batch_traj_buffer(
                buffer=buffer,
                env=env,
                afn=model,
                num_trajectories=cfg.num_regen_traj,
                batch_size=cfg.buffer_batch_size,
            )
            model.train()

    return model
