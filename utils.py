import torch
import os
import json
import sacrebleu
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from data import SeqToSeqEncode
import nltk
from tqdm import *
import logging as log
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import PeftModel, LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import  LoraGAContext, save_loraga_model_init, save_loraga_model_final ,estimate_gradient
from accelerate import Accelerator
from typing import Tuple, List, Dict
import wandb


def collate_fn_with_cuda(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(batch[0], dict):
        return {key: torch.stack([item[key] for item in batch]).to(device) for key in batch[0]}
    elif isinstance(batch[0], (list, tuple)):
        return [torch.stack([item[i] for item in batch]).to(device) for i in range(len(batch[0]))]
    else:
        return torch.stack(batch).to(device)


def initialize_model(
    model_name:str
    ):
    if "mt5" in model_name:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model,tokenizer
    

def model_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_target_length: int = 256,
):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_target_length)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred_text


def train_text_to_text_model(    
    run_name: str,
    input_language:str,
    target_language:str,
    dataset: Dataset,
    #valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    per_device_batch_size: int = 1,
    max_length: int = None,
    training_strategy:str = None,
    **kwargs,):

    eval_steps = 10000
    base_path = os.path.dirname(__file__)
    output_dir = os.path.join(base_path,f"results/{run_name}")

    if input_language=='zh':
        num_epochs = 10 
        lr = 1e-4
    elif input_language == 'ja':
        num_epochs = 3 
        lr = 1e-4
    else:
        num_epochs = 2
        lr = 1e-4

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy = "steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
        remove_unused_columns=False
    )

    append_path = f"{input_language}_{target_language}_{training_strategy}.json"
    write_dir = os.path.join(base_path,f"outs/{append_path}")

    def compute_metrics(pred):
        predictions, labels = pred.predictions, pred.label_ids
        sentence_num = labels.shape[0]
        bleu_score,meteor_score,ter_score = 0,0,0
        sum = 0
        with open(write_dir,'w') as f:
            for i in range(sentence_num): 
                label = np.array(labels[i])
                label = np.maximum(label,0)
                label = label.tolist()

                decoded_pred = tokenizer.decode(predictions[i], skip_special_tokens=True)
                decoded_label = tokenizer.decode(label, skip_special_tokens=True)

                data = {'pred':decoded_pred,'target':decoded_label}
                json_string = json.dumps(data, ensure_ascii=False)
                f.write(json_string)
                f.write('\n')

                word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
                prediction = word_tokenizer.tokenize(decoded_pred)
                label = word_tokenizer.tokenize(decoded_label)

                try:
                    bleu_score += sentence_bleu([label], prediction,smoothing_function=SmoothingFunction().method2 )
                    ter_score += sacrebleu.metrics.TER().corpus_score(prediction, label).score
                    meteor_score += single_meteor_score(label, prediction)
                    sum += 1
                except IndexError:
                    print(f"label:{label},prediction:{prediction}")
                else:
                    pass
            

        bleu_score,meteor_score,ter_score = bleu_score/sum,meteor_score/sum,ter_score/sum
        print(f"bleu:{bleu_score},meteor:{meteor_score},ter:{ter_score}")

        return {
            "bleu": bleu_score,
            "meteor": meteor_score,
            "ter": ter_score,
            "sum": sum
        }
    
    dataset.set_transform(lambda x: SeqToSeqEncode(x,input_language,target_language, tokenizer, max_length))
    train_dataset,valid_dataset = dataset['train'],dataset['test']
    model = model.to("cuda")
    # train_dataset.set_transform(lambda x: SeqToSeqEncode(x,input_language,target_language, tokenizer, max_length))
    # valid_dataset.set_transform(lambda x: SeqToSeqEncode(x,input_language,target_language, tokenizer, max_length))

    if training_strategy == "loraga":
        peft_config = LoraGAConfig(bsz=per_device_batch_size,r=8,target_modules = ["q","k","v","o"])
        # Estimate gradients
        dataloader = DataLoader(dataset['train'].select(range(int(dataset['train'].num_rows/100))), batch_size=16, shuffle=True, collate_fn=collate_fn_with_cuda)
        accelerator = Accelerator()

        named_grad = estimate_gradient(
            model=model,
            dataloader=dataloader,
            accelerator = accelerator,
            #batch_size= 16
        )
        # Use the LoraGAContext to attach named gradients to the model
        with LoraGAContext(model=model, named_grad=named_grad):
            model = get_peft_model(model=model, peft_config=peft_config)
        
        # trainable_params, all_param = model.get_nb_trainable_parameters()
        # print(model,f"\n\ntrainable rate: {trainable_params,trainable_params/all_param,trainable_params/(all_param+trainable_params)}")
        # save_loraga_model_init(model, save_dir=output_dir)
    elif training_strategy == "lora":

        peft_config = LoraConfig(
            r=8, # lora rank
            lora_alpha=32,  
            target_modules=["q", "v", "k", "o"],  #"o_proj, "k_proj""
            lora_dropout=0.1, 
            bias="none"  
        )
        
        model = get_peft_model(model, peft_config)

        # For debug
        model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters() )
    print(model,f"\n\ntrainable rate: {trainable_params,trainable_params/all_params}")
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    if training_strategy == "loraga":
        # Save the final state of the LoRA-GA model
        save_loraga_model_final(model, save_dir=output_dir)
        # Load the saved model like you would load a LoRA model
        model = PeftModel.from_pretrained(model, output_dir)
    else:
        model.save_pretrained(output_dir)
    
    return model

    

if __name__ == "__main__":
    pass
