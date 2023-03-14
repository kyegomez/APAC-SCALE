import time
import torch 
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import get_scheduler, default_data_collator, get_linear_schedule_with_warmup
from torch.optim import AdamW
from lion_pytorch import Lion


from kosmos import Kosmos, KosmosTokenizer
from accelerate import Accelerator


from rich.progres import Progress
from datasets import Image
from bitsandbytes.optim import AdamW8bit


def count_number_of_parameters(model, only_trainable: bool = True) -> int:
    if only_trainable:
        num_param: int = sum(p.numel()
                            for p in model.parameters() if p.requires.grad)
        
    else:
        num_params: int = sum(p.numel() for p in model.parameters() if p)

    

def prep_sample(sample):
    question = sample["question"]
    answer = sample["answer"].split("|!+")[1]
    explanation = sample["explanation"]
    text = f"Question: {question} Answer: {answer} Explanation: {explanation}" 
    image = sample["image"]
    return {
        "image": image,
        "target_text": text
    }


def train(args):
    accelerator = Accelerator(
        mixed_precision="fp16"
    )

    #if passed along set the training seed now
    if args.seed is not None:
        set_seed(args.seed)

    model = Kosmos()
    model = model.to(accelerator.device)


    optimizer = Lion(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    tokenizer = KosmosTokenizer()
    dataset = load_dataset("bjoernp/vqax", split="text")
    #dataset = dataset.cast_column("url", Image)
    dataset = dataset.map(prep_sample, num_proc=8)
    remove_columns = ['id', 'img_id', 'question', 'answer', 'explanation', 'none', 'image', 'target_text']
    dataset = dataset.map(tokenizer.tokenize, batched=True,
                        batch_size=128, remove_columns=remove_columns)
        
    train_dataloader = DataLoader(
            dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
        )
    
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, optimizer, lr_scheduler)

    model.train()
    accelerator.register_for_checkpointing(lr_scheduler)

    model.clip_model.requires_grad_(False)
    model.clip_model.encoder.layers[-1].requires_grad_(True)


    accelerator.print(f"number of parameters: {count_number_of_parameters(model):,}")
    accelerator.print(f"number of trainable parameters: {count_number_of_parameters(model, only_trainable=True):,}")

    #log model and optimizer paramneters to wandb
    accelerator.init_trackers(project_name="kosmos")

    train_loader = iter(train_dataloader)
    epoch_loss=0
    total_loss=0
    start_time = time.time()


    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=args.max_steps)
        for step in range(0, args.max_steps):
            batch_start = time.time()
            batch = next(train_loader)
            outputs = model(**batch, self_attn_padding_mask=batch["attention_mask"])
            #shift so that tokens < n predict n
            outputs = torch.cat([outputs[:, :1], outputs[:, 67:]], dim=1).contigous()
            #shift_logits = outputs[..., :-1, :].contigous()
            # shift_labels=batch["labels"][..., 1:].contigous()
            #flatten the tokens
            loss_fct = CrossEntropyLoss()
            one_hot_labels = torch.nn.functional.one_hot(batch["labels"][:, 1:], num_classes=32002).float()
            loss = loss_fct(outputs[:, :-1], one_hot_labels)

            epoch_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            batch_end = time.time()
            logs = {
                "loss": loss.items(),
                "perplexity": torch.exp(loss).item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "examples": args.batch_size * (step + 1),
                "examples_per_second": args.batch_size / (batch_end - batch_start),
            }
            if step % args.log_every == args.log_every - 1:
                accelerator.log(logs, step=step)
                progress.update(task, advance=1, description=f"Step Loss: {loss.item():.5f} "
                                                             f"| Mean Loss: {(total_loss + epoch_loss) / step:.5f} "
                                                             f"| Mean PPL: {torch.exp((total_loss + epoch_loss) / step):.2f} "
                                                             f"| Examples: {args.batch_size * (step + 1)} "
                                                             f"| Examples/s: {args.batch_size / (batch_end - batch_start):.2f} "
                                                             f"| Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

            if step % args.save_every == args.save_every - 1:
                train_epoch_loss = epoch_loss / args.save_every
                total_loss += epoch_loss
                epoch_loss = 0

                accelerator.log({
                    "train_ppl": torch.exp(train_epoch_loss),
                    "train_epoch_loss": train_epoch_loss,
                }, step=step)

                progress.print(f"Saving checkpoint at step {step}...")
                accelerator.save_state(
                    f"{args.checkpoint_dir}/checkpoint_at_step_{step}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train(args)