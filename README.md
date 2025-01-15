# **Extending Multi-Language Translation with PEFT and LLM Evaluation**: Final Project for NLPDL

# File Description

- **data.py**: Contains functions to load all tatoeba datasets. The corresponding datasets need to be downloaded and placed in the `data` folder. Details see [Dataset Preparation](#dataset-preparation)
- **llm_eval.py**: Tests the output using our implemented LLM-Eval.
- **train.py**: Training script, where hyperparameters can be modified.
- **utils.py**: Contains various functions needed during training, testing, etc., mainly including code for using LoRA-GA and lora.
- **config.yaml**: Contains model name, source language, target language, etc. Modify the config to train translation models in different languages.
- **requirements.txt**: Dependencies.
- **data**: Folder for storing datasets.
- **peft**: PEFT library required for LoRA-GA.
- **outputs**: Stores translation outputs and standard outputs, used for LLM-Eval.
- **checkpoints**: Stores model checkpoints during training. Not necessarily needed for the LLM-Evaluation in this code.



# Environment Preparation
Our code is built based on:
- CUDA 11.8
- Python 3.10  

To set up the required environment for this code, first install the requirements by:
```bash
pip install -r requirements.txt
```

One possible issue you may encounter is being unable to install the `nltk` package, which can be fixed by trying the following:
   
```bash
python
import nltk
nltk.download('wordnet')
nltk.download('punkt')  
nltk.download('omw-1.4')    
```

Next, unzip and install the peft code pack:
 ```bash
unzip peft.zip
pip install -e peft
```

# Dataset Preparation
We conduct our experiment on the Tatoeba dataset. To prepare this dataset, please download the corresponding languages(German, Japanese and Chinese) from https://tatoeba.org/zh-cn/downloads. Find the *Custom Exports* label, choose source language and target language and download. The downloaded `.tsv` files should be put in `./data/tatoeba` folder.

# Run Our Code



## Training 
The training and evaluation with traditional metrics of our project can be reproduced by:
```bash
python train.py input_language='de' target_language='en' training_strategy='loraga'
```
You can select `input_language` from `['de','ja','zh']`, `target_language`(we only implemented `'en'`), and `training_strategy` from `['full fine-tune','lora','loraga']` to perform different traing experiments.
You can also modify these settings along with the backbone settings in `config.yaml`. Other hyper-parameters canbe tuned in `train.py`.

This script will directly output the traditional metrics and save the trained models in `./checkpoints`, the translation results in `./outs` which could be used for the LLM-Evaluation in the next step.

## LLM-Evaluation
Before running the following command, please make sure that you have already prepared the `<in_lan>_<out_lan>_<metho>.json` files in the `outputs` folder.


 ```bash
python llm_eval.py
```
You can modify the input to the `main` function in `llm_eval.py` to select the language for testing. You also need replace the `openai_api_key` to your own api keys.

