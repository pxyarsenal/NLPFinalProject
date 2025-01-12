# File Description

- **data.py**: Contains functions to load all datasets. The corresponding datasets need to be downloaded and placed in the `data` folder.
- **llm_eval.py**: Tests the output using BLEU, METEOR, and LLM-Eval.
- **train.py**: Training script, where hyperparameters can be modified.
- **utils.py**: Contains various functions needed during training, testing, etc., mainly including code for using LoRA-GA.
- **config.yaml**: Contains model name, source language, target language, etc. Modify the config to train translation models in different languages.
- **requirements.txt**: Dependencies.
- **data**: Folder for storing datasets.
- **peft**: PEFT library required for LoRA-GA.
- **outs**: Stores evaluation outputs and standard outputs, used for LLM-Eval.

# Reproduction Instructions

1. You need to download the dataset yourself. The dataset used is Tatoeba.
2. Set up the environment:
    ```bash
    cd NLPFinalProject-main
    unzip peft.zip
    pip install -r requirements.txt
    pip install -e peft
    ```
3. Conduct experiments:
    ```bash
    python train.py input_language='de' target_language='en' training_strategy='loraga'
    ```
    - You can set `input_language`, `target_language`, and `training_strategy` to perform different experiments.

    ```bash
    python llm_eval.py
    ```
    - You can modify the input to the `main` function in `llm_eval.py` to select the language for testing.

# Additional Environment Information

- CUDA 11.8
- Python 3.10
