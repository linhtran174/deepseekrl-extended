"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from shutil import copyfile
from collections import defaultdict
from qwen_vl_utils import process_vision_info

from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig

import llms
import utils
import evaluator
import rldatasets

def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set with improved logging and PDF reports.
    (Orchestrates evaluation using helper functions).
    """
    print("Running evaluation on test set...")
    
    # --- Initialization ---
    num_examples = 0
    total_correct = 0
    num_total_chains_processed = 0 
    aggregated_metrics = defaultdict(float)
    
    _, pdf_dir, json_dir = utils._setup_eval_directories(args.output_dir)
    pdf_path = os.path.join(pdf_dir, f'eval_results_{round_num}.pdf')
    doc, styles, story = utils._setup_pdf(pdf_path)
    
    test_loader.reset()

    # --- Evaluation Loop ---
    for batch in tqdm(test_loader, desc="Evaluating on test set"):
        img_path, answer = batch
        
        # Generate
        _, _, _, _, completions_text, _ = generate_completions(
            model, tokenizer, img_path, test_loader.prompt, device, args, eval=True)
        
        # Add example header to PDF
        utils._add_example_header_to_pdf(story, styles, img_path, test_loader.prompt, answer, num_examples + 1)
        

        # Process each completion
        example_total_correct = 0
        example_num_chains = 0
        for completion_idx, completion_text in enumerate(completions_text):
            metrics_single, num_correct, num_total = utils._process_single_completion_for_eval(
                completion_text, eval_class, answer, device, story, styles, completion_idx
            )
            
            # Accumulate metrics
            if metrics_single and isinstance(metrics_single, dict):
                for metric_name, value in metrics_single.items():
                    if isinstance(value, (int, float)): 
                        aggregated_metrics[metric_name] += value
                    elif torch.is_tensor(value) and value.numel() == 1:
                         aggregated_metrics[metric_name] += value.item()
            
            example_total_correct += num_correct
            example_num_chains += num_total

        total_correct += example_total_correct
        num_total_chains_processed += example_num_chains

        num_examples += 1


    # --- Finalization ---
    try:
        doc.build(story)
        print(f"Evaluation PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

    avg_scores, accuracy = utils._calculate_and_log_final_metrics(
        aggregated_metrics, total_correct, num_total_chains_processed, 
        num_examples, json_dir, round_num, args.verbose
    )

    return avg_scores, accuracy

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    image_path: str,
    prompt: str,
    device: str,
    args: argparse.Namespace, 
    eval: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        image_path: The input image path to generate completions for
        prompt: The input prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """

    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. You are an expert image analyst.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}

            ],
        },
    ]

    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)  
    image_inputs, video_inputs = process_vision_info(conversation)

    # Ensure left padding for tokenizer/processor before tokenizing
    prompt_inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device).to(model.dtype)

    # Repeat input tensors for batch generation
    if eval:
        num_chains = args.num_chains_eval
    else:
        num_chains = args.num_chains
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(num_chains, *([1] * (value.dim() - 1)))
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value 

    # Original prompt_ids/mask are needed for splitting later
    original_prompt_ids = prompt_inputs["input_ids"]

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
    )

    # Generate all completions at once
    prompt_completion_ids = model.generate(
        **batched_prompt_inputs,
        generation_config=generation_config

    )

    
    # Extract completion ids
    # Use the original prompt length before repeating
    prompt_length = original_prompt_ids.size(1) 
    prompt_ids = prompt_completion_ids[:, :prompt_length] # These are the batched prompt IDs
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
    is_eos = completion_ids == tokenizer.tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # Create attention mask based on original prompt mask repeated and completion mask
    prompt_mask = batched_prompt_inputs["attention_mask"] # Use the repeated mask
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt
    
def score_completions(
    completions_text: list[str],
    prompt: str,
    image_path: str,
    answer: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': prompt,
            'answer': answer, 
            'image_path': image_path
        },
        'generations': []
    }
    
    # Format inputs as expected by evaluator
    mock_completions = [[{'content': completion}] for completion in completions_text]
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=None, completions=mock_completions, answer=answer, device=device
    )

    rewards = rewards_per_func.sum(dim=1)

    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data

def compute_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel, 
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace,
    img_path: str,
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits
    logits_to_keep = completion_ids.size(1)

    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = utils.get_per_token_logps_vl(base_model, prompt_completion_ids, attention_mask, img_path, tokenizer, logits_to_keep, prompt)

    # Get training model logits
    per_token_logps = utils.get_per_token_logps_vl(model, prompt_completion_ids, attention_mask, img_path, tokenizer, logits_to_keep, prompt)

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # Compute loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    metrics["kl"] = mean_kl.item()

    return loss, metrics


def grpo_loss(
        model: PreTrainedModel,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        img_path: str,
        answer: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        tokenizer: Tokenizer for the models
        prompt: Prompt for the model
        img_path: Path to the image
        answer: Ground truth answer
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """



    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
        model, tokenizer, img_path, prompt, device, args
    )


    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, prompt, img_path, answer, eval_class, device, args
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)
    image_log_dir = os.path.join(training_log_dir, 'images')
    os.makedirs(image_log_dir, exist_ok=True)
    image_log_path = os.path.join(image_log_dir, f'image_{round_num}.png')
    copyfile(img_path, image_log_path)


    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args, img_path, tokenizer, prompt
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Name/path of base model")
    parser.add_argument("--dataset_name", type=str, default="flowers", help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="flowers", help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a resumable checkpoint every N steps")
    parser.add_argument("--eval_iterations", type=int, default=100, help="Number of iterations for evaluation")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training from.")

    # Optimization hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Percentage of total steps for warmup")
    parser.add_argument("--update_ref_model", action="store_true", help="Whether to update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="How often to update reference model")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, help="Alpha parameter for reference model mixup")


    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=16, help="Number of parallel generation chains")
    parser.add_argument("--num_chains_eval", type=int, default=2, help="Number of parallel generation chains for evaluation")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=3000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 
    
    # Seed everything 
    utils.seed_everything(args.seed)

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    ###############################
    ## Main Experiment settings ##
    ###############################

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.evaluator)

    ###############################


    # Setup logging 
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Only save args if not resuming (or if resuming and args.json doesn't exist)
    args_path = os.path.join(args.output_dir, 'args.json')
    if not args.resume_from_checkpoint or not os.path.exists(args_path):
        args_dict = vars(args)
        with open(args_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
            
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)


    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)
    
    start_round = 0
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Load base model state if it exists in the checkpoint (optional, depends if you also save/update it)
            if 'base_model_state_dict' in checkpoint:
                 base_model.load_state_dict(checkpoint['base_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_round = checkpoint['round_num'] + 1 # Start from the next round
            print(f"Loaded checkpoint. Resuming from round {start_round}")
            temp_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step + start_round))
            scheduler = temp_scheduler 
        else:
            print(f"Warning: Checkpoint file not found at {args.resume_from_checkpoint}. Starting training from scratch.")


    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    if os.path.exists(os.path.join(train_log_dir, "train_logs.json")):
         with open(os.path.join(train_log_dir, "train_logs.json"), "r") as f:
            # Load existing logs if resuming
            try:
                train_metrics_total = json.load(f)
                 # Convert string keys back to integers if necessary
                train_metrics_total = {int(k): v for k, v in train_metrics_total.items()}
            except json.JSONDecodeError:
                 print("Warning: Could not parse existing train_logs.json. Starting with empty logs.")
                 train_metrics_total = {}

    for round_num in tqdm(range(start_round, args.num_train_iters), initial=start_round, total=args.num_train_iters, desc="Training Progress"):
    
        # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

        # Slowly update ref model
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get next question
        batch = next(train_loader)
        img_path, answer = batch

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(model, base_model, tokenizer, train_loader.prompt, img_path, answer, eval_class, device, round_num, train_log_dir, args)
        # Gradient accumulation
        total_loss = total_loss # / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    

        # Save checkpoint
        if (round_num + 1) % args.save_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_round_{round_num}.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'base_model_state_dict': base_model.state_dict(), # Save base model state too
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args # Save args for reference
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() # * args.gradient_accumulation_steps - Loss is already averaged over accumulation steps before backward
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)

        # Add after each major operation in the training loop
        torch.cuda.empty_cache()
       
