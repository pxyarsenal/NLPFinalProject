import hydra
import torch
from omegaconf import DictConfig
from data import DATASET_MAP
from utils import train_text_to_text_model, initialize_model
import wandb

@hydra.main(config_name="config")
def main(cfg):
    print(cfg)
    torch.cuda.reset_peak_memory_stats()
    model_name_or_path = cfg.model_name_or_path
    input_language = cfg.input_language
    target_language = cfg.target_language
    run_name = cfg.run_name
    training_strategy = cfg.training_strategy
    dataset_n = cfg.dataset_name

    try:
        wandb.init(
        project="NLP_FINAL",
        name= f"{input_language}_{target_language}_{training_strategy}"
        )
    except Exception:
        print("Disabled Wandb visualization.")


    run_name += "_" + input_language + "_" + target_language+ "_" +training_strategy +'_'+dataset_n
    model, tokenizer = initialize_model(model_name_or_path)
    #print(model)
    if dataset_n == "tatoeba":
        dataset_name =f"tatoeba_{min(input_language,target_language)}_{max(input_language,target_language)}"  
    else:
        dataset_name = f"news_commentary_{min(input_language,target_language)}_{max(input_language,target_language)}"
    dataset = DATASET_MAP[dataset_name]()
    model = train_text_to_text_model(run_name,input_language,target_language,dataset,model,tokenizer,16,128,training_strategy)
    print(f"GPU max allocated: {torch.cuda.max_memory_allocated() / 1024**2}MB")

    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()