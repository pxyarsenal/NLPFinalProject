import logging
import json
import re
import os
import hydra
from tqdm import *
from typing import Dict, List
from data import DATASET_MAP

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic, OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import create_model

logging.basicConfig(level=logging.INFO)

turn_eval_template = (
    "{format_instructions}\n"
    "Score the following dialogue response generated on a continuous scale from {score_min} to {score_max}.\n"
    "Context: {context}\n"
    "Reference: {reference}\n"
    "Dialogue response: {response}"
)

turn_noref_eval_template = (
    "{format_instructions}\n"
    "Score the following dialogue response generated on a continuous scale from {score_min} to {score_max}.\n"
    "Context: {context}\n"
    "Dialogue response: {response}"
)

dialogue_eval_template = (
    "{format_instructions}\n"
    "Score all the following response generated by the model in a multi-turn dialogue on a continuous scale from {score_min} to {score_max}.\n"
    "Dialogue: {dialog}"
)

score_config = {
    "0-5": {
        "score_min": 0.0,
        "score_max": 5.0,
        "score_dtype": float,
    },
    "0-100": {
        "score_min": 0,
        "score_max": 100,
        "score_dtype": int,
    },
}


def generate_score_model(
    field_names: Dict[str, str], score_type: type, score_range: tuple
) -> type:
    fields = {}
    for field_name in field_names:
        fields[field_name] = (score_type, ...)

    ScoreModel = create_model("ScoreModel", **fields)

    for field_name, field_info in ScoreModel.__fields__.items():
        field_info.field_info.description = f"{field_names[field_name]} score in the range of {score_range[0]} to {score_range[1]}"

    return ScoreModel


def get_pydantic_output_parser(*args, **kwargs) -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=generate_score_model(*args, **kwargs))


def run_eval_chain(
    score_aspects: List[str],
    score_dtype: type,
    score_min: float,
    score_max: float,
    human_template: str,
    model_name: str = "gpt-3.5-turbo",
    **prompt_kwargs,
):
    if "gpt-3.5" in model_name:
        chat = ChatOpenAI(temperature=0, model_name=model_name, max_retries=0,openai_api_base="https://chatapi.littlewheat.com/v1",openai_api_key="sk-GKIo93w3qdDnbOsXCBueXh6h49oSyfksAZuVX3JXOYglxpsC")
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    elif "text-davinci" in model_name:
        chat = OpenAI(model_name=model_name, temperature=0)
        input_variables = re.findall(r"\{(\w+)\}", human_template)
        chat_prompt = PromptTemplate(
            template=human_template, input_variables=input_variables
        )
    elif "claude" in model_name:
        chat = Anthropic(model=model_name, temperature=0)
        input_variables = re.findall(r"\{(\w+)\}", human_template)
        chat_prompt = PromptTemplate(
            template=human_template, input_variables=input_variables
        )
    else:
        raise ValueError("Unknown model name %s" % model_name)

    chain = LLMChain(
        prompt=chat_prompt,
        llm=chat,
        verbose=True,
    )

    parser = get_pydantic_output_parser(
        field_names={aspect: aspect for aspect in score_aspects},
        score_type=score_dtype,
        score_range=(score_min, score_max),
    )
    try:
        output = chain.run(
            format_instructions=parser.get_format_instructions(),
            score_min=score_min,
            score_max=score_max,
            **prompt_kwargs,
        )
        scores = parser.parse(output)
    except Exception as e:
        logging.warning("Failed to run chain: %s" % e)
        return None, None

    return output, scores


def main(in_lang,tar_lang,filepath, outpath):
    model_name = "gpt-3.5-turbo"
    aspects = ["fluency", "coherence", "relevance", "engagingness"]
    # os.environ["OPENAI_API_KEY"] = "sk-x13OeuPSApGK5zsbQaToT9ycxSqCuJzgVx8WVfRIJ3hBBr7b"
    dataset_n = f"tatoeba_{min(in_lang,tar_lang)}_{max(in_lang,tar_lang)}"
    dataset = DATASET_MAP[dataset_n]()['test']
    filename = filepath
    total_score = [0,0,0,0]
    sum = 0

    outs = open(outpath,"w")
    index = 0
    max_num = 1000

    with open(filename, "r") as f:
        t = tqdm(f.readlines())
        for line in t:
            max_num -= 1
            data = json.loads(line)
            # while dataset[index][tar_lang] != data["target"]:
            #     index += 1
            context = dataset[index][in_lang]
            response = data["pred"]
            reference = data["target"]

            raw_output, scores = run_eval_chain(
                model_name=model_name,
                score_aspects=aspects,
                human_template=turn_eval_template,
                context=context,
                response=response,
                reference=reference,
                **score_config["0-5"],
            )
            #print(f"Raw output: {raw_output}")
            index += 1
            print(f"Scores: {scores}",file=outs)
            try:
                total_score[0] += scores.fluency
                total_score[1] += scores.coherence
                total_score[2] += scores.relevance
                total_score[3] += scores.engagingness
                sum +=1
            except AttributeError:
                print("something wrong")
            else:
                print("something wrong")
            t.set_description(f"AVG_FLUENCY: {total_score[0]/sum:05f},AVG_COHERENCE: {total_score[1]/sum:05f},AVG_RELEVANCE: {total_score[2]/sum:05f},AVG_ENGAGINGNESS: {total_score[3]/sum:05f},SUM_EXAMPLES:{sum}")
            if max_num == 0:
                break

    print(f"AVG_FLUENCY: {total_score[0]/sum:05f},AVG_COHERENCE: {total_score[1]/sum:05f},AVG_RELEVANCE: {total_score[2]/sum:05f},AVG_ENGAGINGNESS: {total_score[3]/sum:05f} ",file=outs)

if __name__ == "__main__":
    main('zh','en',"/ceph/home/penshenyao/NLP/final/outs/zh_en_fine_tuning.json","/ceph/home/penshenyao/NLP/final/outs_llm/en_zh_finetune.txt")