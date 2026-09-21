import os
import time

import torch

from src.args import parse_arguments
from src.attention_ft import configure_attention_finetuning
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.mergopt import DEFAULT_MERGOPT_K_MAX, MergOPT
from src.modeling import ImageClassifier, ImageEncoder
from src.result_names import finetuned_checkpoint_name, training_checkpoint_name
from src.sam import SAM
from src.utils import LabelSmoothing, cosine_lr


# Step-controlled fine-tuning. All tasks train for args.num_steps and save checkpoints.
# Examples:
# python -m src.indep_finetune --finetuning-mode=standard --model=ViT-B-32
# python -m src.indep_finetune --finetuning-mode=linear --model=ViT-B-32
# python -m src.indep_finetune --finetuning-mode=attention --model=ViT-B-32
# python -m src.indep_finetune --finetuning-mode=mergopt --model=ViT-B-32
# python -m src.indep_finetune --finetuning-mode=saft --model=ViT-B-32
def _clip_gradients(params, args):
    if args.clip_mode == "noclip":
        return
    if args.clip_mode != "indept":
        raise ValueError("indep_finetune.py supports --clip-mode indept or noclip.")
    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)


def _backward_microbatches(model, microbatches, loss_fn, args):
    loss_value = None
    for inputs, labels in microbatches:
        loss = loss_fn(model(inputs), labels)
        (loss / args.num_grad_accumulation).backward()
        loss_value = loss.detach() if loss_value is None else loss_value + loss.detach()
    return loss_value / len(microbatches)


def finetune(args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "linear",
        "saft",
        "standard",
        "attention",
        "mergopt",
    ], "Only linear, standard, saft, attention, and mergopt are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    saft_finetuning = args.finetuning_mode == "saft"
    attention_finetuning = args.finetuning_mode == "attention"
    mergopt_finetuning = args.finetuning_mode == "mergopt"
    run_name = args.run_name or None
    if linearized_finetuning:
        print("Using linearized fine-tuning.")
    if saft_finetuning:
        print(f"Using SAFT (ASAM) fine-tuning with rho={args.saft_rho}.")
    if attention_finetuning:
        print("Using attention-only fine-tuning.")
    if mergopt_finetuning:
        if args.mergopt_k_max is None:
            args.mergopt_k_max = DEFAULT_MERGOPT_K_MAX
        print(
            "Using MergOPT fine-tuning "
            f"(mu={args.mergopt_mu}, b={args.mergopt_b}, "
            f"k_max={args.mergopt_k_max})."
        )

    # Check if checkpoints already exist (if exist then return)
    ft_path = os.path.join(
        ckpdir,
        finetuned_checkpoint_name(args.finetuning_mode, run_name),
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "zeroshot.pt")
    )
    # Skipping if already exist
    # if os.path.exists(zs_path) and os.path.exists(ft_path):
    #     print(f"Skipping fine-tuning because {ft_path} exists.")
    #     return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    # If a checkpoint path is provided, load an existing encoder.
    # Otherwise, build one from scratch.
    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (         # This is building by using an openai pretrained model
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.to(args.device)
    attention_params = (
        configure_attention_finetuning(model) if attention_finetuning else None
    )

    preprocess_fn = model.train_preprocess
    print_every = 100
    # Return the dataset class from dataset_name.
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    # Return train or test dataloader
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    # Can use different loss functions
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = attention_params or [p for p in model.parameters() if p.requires_grad]
    if saft_finetuning:
        optimizer = SAM(
            params,
            torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.wd,
            rho=args.saft_rho,
            adaptive=True,
        )
    elif mergopt_finetuning:
        optimizer = MergOPT(
            params,
            torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.wd,
            mu=args.mergopt_mu,
            b=args.mergopt_b,
            k_max=args.mergopt_k_max,
            alphas=args.mergopt_alphas,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer.base_optimizer if saft_finetuning else optimizer,
        args.lr,
        args.warmup_length,
        args.num_steps,
    )

    # Saving zero-shot model
    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        model.image_encoder.save(model_path)

    model.train()
    optimizer.zero_grad()
    epoch = 0
    batch_idx = 0
    data_iter = iter(data_loader)

    for step in range(args.num_steps):
        step_loss = None
        microbatches = []
        # Accumulate batches according to num_grad_accumulation
        for _ in range(args.num_grad_accumulation):
            start_time = time.time()
            # If the dataloader runs out, restart it and count that as a new epoch.
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                batch_idx = 0
                data_iter = iter(data_loader)
                batch = next(data_iter)

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(args.device)
            labels = batch["labels"].to(args.device)
            microbatches.append((inputs, labels))
            data_time = time.time() - start_time

            if not mergopt_finetuning:
                logits = model(inputs)

                loss = loss_fn(logits, labels)
                step_loss = loss

                (loss / args.num_grad_accumulation).backward()

            batch_time = time.time() - start_time
            batch_idx += 1

        scheduler(step)

        if mergopt_finetuning:
            def closure(microbatches=microbatches):
                optimizer.zero_grad()
                closure_loss = _backward_microbatches(
                    model,
                    microbatches,
                    loss_fn,
                    args,
                )
                _clip_gradients(params, args)
                return closure_loss

            step_loss = optimizer.step(closure)
        else:
            _clip_gradients(params, args)

        if saft_finetuning:
            def closure(microbatches=microbatches):
                closure_loss = _backward_microbatches(
                    model,
                    microbatches,
                    loss_fn,
                    args,
                )
                return closure_loss

            optimizer.step(closure)
        elif not mergopt_finetuning:
            optimizer.step()
        optimizer.zero_grad()

        current_step = step + 1

        # Save checkpoints according to args.checkpoint_every.
        if args.checkpoint_every > 0 and current_step % args.checkpoint_every == 0:
            print("Saving checkpoint.")
            model_path = os.path.join(
                ckpdir,
                training_checkpoint_name(
                    args.finetuning_mode,
                    current_step,
                    run_name,
                ),
            )
            model.image_encoder.save(model_path)

        if current_step == 1 or current_step % print_every == 0:
            percent_complete = 100 * current_step / args.num_steps
            print(
                f"Train Step: {current_step}/{args.num_steps} [{percent_complete:.0f}%]\t"  # noqa: E501
                f"Epoch: {epoch}\tBatch: {batch_idx}/{num_batches}\t"
                f"Loss: {step_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                flush=True,
            )

    image_encoder = model.image_encoder
    # eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None:
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ft_path = os.path.join(
            ckpdir,
            finetuned_checkpoint_name(args.finetuning_mode, run_name),
        )
        image_encoder.save(ft_path)
        return zs_path, ft_path


if __name__ == "__main__":
    DEFAULT_DATASETS = [
        "CIFAR100",
        "Flowers102",
        "PCAM",
        "FER2013",
        "Cars",           # 7,330
        "DTD",            # 1,692
        # "EuroSAT",        # 16,200
        "GTSRB",            # 35,289
        # "MNIST",            # 55,000
        "RESISC45",       # 17,010
        "SUN397",         # 17,865
        "SVHN",           # 68,257
    ]
    args = parse_arguments()
    train_datasets = args.train_dataset or DEFAULT_DATASETS
    if args.save is None:
        args.save = (
            f"checkpoints_{args.seed}/{args.model}"
            if args.seed is not None
            else f"checkpoints/{args.model}"
        )
    if args.finetuning_mode == "mergopt" and args.mergopt_k_max is None:
        args.mergopt_k_max = len(train_datasets)
    for dataset in train_datasets:

        assert args.num_steps > 0, "--num-steps must be positive."

        args.train_dataset = dataset + "Val"

        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset} for {args.num_steps} steps")
        print("=" * 100)
        finetune(args)
