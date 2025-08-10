import os, os.path as osp
import json
import argparse
import torch
from llava import conversation as conversation_lib
from utils.data_collator import CustomDataCollator
from PIL import Image
import numpy as np

from posegpt.utils import Config

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from scripts.instructions.finetune_instructions import instructions as reasoning_template, action_template
from utils.load import load_unipose_model_4bit


def main(args):
    # disable_torch_init()
    config = Config.fromfile(args.config)
    torch_dtype = torch.bfloat16
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f'cuda:{local_rank}'
    conversation_lib.default_conversation = conversation_lib.conv_templates['mistral_instruct']

    # build model, tokenizer
    print('Load model...')
    model, image_processor = load_unipose_model_4bit(
        config, 'cache/unipose_merged', torch_dtype=torch_dtype, device_map={"": local_rank}, **config)

    model.config.use_cache = False # silence the warnings
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    tokenizer = model.tokenizer

    with open("data/trainable/target_modules.json") as f:
        target_modules = json.load(f)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    if local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} with shape {tuple(param.shape)}")

    dataset = load_dataset("Jucha/cot-poses-data", split="train")
    pose_begin_token_id = 34048
    pose_end_token_id = 34049
    pose_query_begin_token_id = 34052
    pose_query_end_token_id = 34052 + 80
    def preprocess(example):
        action = example["action"]
        reasoning = example["reasoning_refined"] if example["reasoning_refined"] not in [None, ""] else example["reasoning"]
        answer = example["answer"]
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []

        instruction = action_template.replace('<action>', action)
        reasoning_base = np.random.choice(reasoning_template.text2pose['input'])
        reasoning = reasoning_base.replace('<caption>', reasoning)
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], reasoning + ' ' + answer)

        toks = tokenizer(
            conv.get_prompt(),
            truncation=True,
            max_length=512,
            padding='max_length',
        )

        labels = toks["input_ids"].copy()
        labels = [lab if am==1 else -100
                for lab, am in zip(labels, toks["attention_mask"])]
        toks["labels"] = labels

        ### Pose token between <pose_id_2048> and <pose_id_2049>
        ### input_ids should be <pose_query_0> (pose_query_begin_token_id)
        ### to <pose_query_79> (pose_query_end_token_id)
        ### attention mask for pose token should be 2
        pose_begin_idx = toks["input_ids"].index(pose_begin_token_id)
        pose_end_idx = toks["input_ids"].index(pose_end_token_id)
        input_ids = toks['input_ids'].copy()
        input_ids[pose_begin_idx+1:pose_end_idx] = list(range(pose_query_begin_token_id, pose_query_end_token_id))
        toks['input_ids'] = input_ids
        attention_mask = toks["attention_mask"].copy()
        attention_mask[pose_begin_idx+1:pose_end_idx] = [2]*80
        toks["attention_mask"] = attention_mask
        return toks

    train_ds = dataset.map(
        preprocess,
        batched=False,
        remove_columns=dataset.column_names,
    )

    model_name = 'full'
    output_dir = os.path.join("cache", "cot-pose", model_name)
    logdir_dir = os.path.join("logs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logdir_dir, exist_ok=True)

    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=512,
        report_to="tensorboard",
        logging_dir=logdir_dir,
        logging_steps=50,
        logging_first_step=True,
        num_train_epochs=5,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        optim='adamw_torch',
    )

    trainer = SFTTrainer(
        model=model,
        data_collator=CustomDataCollator(tokenizer=tokenizer, mlm=False),
        train_dataset=train_ds,
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    trainer.train()
    trainer.model.save_pretrained(training_arguments.output_dir)

    if trainer.is_world_process_zero():
        model.config.save_pretrained(training_arguments.output_dir)
        with open(osp.join(training_arguments.output_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
        config_dict.pop("quantization_config", None)
        with open(osp.join(training_arguments.output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='cache/unipose')
    parser.add_argument("--model-base", type=str, default='cache/llava-v1.6-mistral-7b')
    parser.add_argument("--config", type=str, default='configs/inference.py')
    args = parser.parse_args()

    main(args)
