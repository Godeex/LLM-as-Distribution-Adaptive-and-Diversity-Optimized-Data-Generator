import random
import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
from datetime import datetime
import re 
import time
from utils import load_attributes, load_attributes_names, load_attributes_names_filter, load_attributes_weights_list, \
    random_read_jsonl_index, read_first_n_jsonl, prompt_prob
import numpy as np
import json


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float, help="which temperature to use")
parser.add_argument("--top_p", default=0.95, type=float, help="which Top-P value to use")
parser.add_argument("--n_sample", default=10, type=int, help="the number of examples generated for each class")
parser.add_argument("--batch_size", default=20, type=int, help="")
parser.add_argument("--dataset", default='amazon', type=str, help="which dataset to use")
parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='dado', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=2048, type=int, help="which seed to use")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")
parser.add_argument("--sample_dir", default='./dataset/amazon/train/train-amazon-choice.jsonl', type=str, help="the folder for saving the example text")

args = parser.parse_args()
print(args.model_name)

api_key = ' '
base_url = ' '


args.api_key = api_key
args.base_url = base_url

if args.dataset in ['sst-2']:
    args.domain = 'movie review'
    args.attributes = ["length", "style", "logic", "emotion"]
elif args.dataset in ['amazon']:
    args.domain = 'review'
    args.attributes = ["product_name", "similar", "length", "style", "logic", "emotion"]
elif args.dataset in ['reddit']:
    args.domain = 'web forum'
    args.attributes = ["similar", "length", "style", "logic", "emotion"]
else:
    raise NotImplementedError


def gen_example(attr_dict):
    """
    Receives a dictionary whose keys are attribute names and values are lists of corresponding values (or iterable objects).
    Returns a generator that generates a dictionary with random indices on each iteration.

    Args:
    attr_dict (dict): Keys are attribute names and values are lists of attribute values (or iterable objects).

    Return:
    generator: Returns a dictionary on each iteration, whose keys are the keys of attr_dict
    and whose values are randomly selected indices (integers) from the corresponding lists.
    """
    lengths = {}
    for x in attr_dict:
        # Traverse each key of attr_dict, calculate the length of the corresponding value, and store it in the lengths dictionary
        lengths[x] = len(attr_dict[x])
    while True:
        return_dict = {}
        for z in lengths:
            # Generates a random integer index between 0 and lengths[z] (not including lengths[z]).
            idx_z = np.random.randint(low=0, high=lengths[z], dtype=int)
            # Add the random index to the return_dict with key z and value idx_z
            return_dict[z] = idx_z 
            # lst.append(return_dict)
        yield return_dict


def dispatch_openai_request(
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> Dict[str, Any]:
    """Dispatches a single request to OpenAI API synchronously.

    This function is intended to be used within an asyncio context but performs
    a synchronous HTTP request to the OpenAI API. For true asynchronicity, you
    would need to use an asynchronous HTTP client library.

    Args:
        messages: List of messages to be sent to OpenAI Chat API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        A response from OpenAI API.
    """
    config = {
        'base_url': args.base_url,
        'api_key': args.api_key
    }
    client = openai.Client(**config)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return response


async def dispatch_openai_requests_async(
        messages_list: List[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> List[Dict[str, Any]]:
    """Dispatches multiple requests to OpenAI API asynchronously using asyncio.

    Args:
        messages_list: List of lists of messages to be sent to OpenAI Chat API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    tasks = [
        asyncio.to_thread(dispatch_openai_request, msgs, model, temperature, max_tokens, top_p)
        for msgs in messages_list
    ]
    return await asyncio.gather(*tasks)


def call_api_async(msg_lst, model, temperature, max_tokens):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total, t= {temperature}.")
    # l = len(msg_lst)

    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        dispatch_openai_requests_async(
            messages_list=msg_lst,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
    )
    # Extract the generated text from the responses
    ans = [x.choices[0].message.content for x in response]
    # ans = [x['choices'][0]['message']['content'] for x in response]

    print(f"API returns {len(ans)} in total.")
    print("===================================")
    return ans


def main(args):
    with open(f"./datasets/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().replace(" ", "_").strip('\n') for x in f.readlines()]
    print(label_names)
    # model = args.model_name
    # openai.api_key = args.api_key
    
    attr_dict = {}
    for attr in args.attributes:
        attr_name = attr
        attr_dict[attr] = load_attributes(
            attr_name=attr_name,
            model=args.model_name,
            dataset=args.dataset,
            method=args.model_type,
            classes=label_names
        )
    # 返回dict -> class: [attr_list]
    attr_name_dict = load_attributes_names(
        model=args.model_name,
        dataset=args.dataset,
        method=args.model_type,
        classes=label_names
    )

    for i, class_name in tqdm(enumerate(label_names)):
        print(i, class_name)
        print(f"Prompt: {args.domain} about {re.sub('_', ' ', class_name)} following...")
        sent_cnt = 0
        attempt = 0
        attr_dict_cls = {}
        for attr in attr_dict:
            # Preset attribute
            if isinstance(attr_dict[attr], list):
                attr_dict_cls[attr] = attr_dict[attr]
            else:
                attr_dict_cls[attr] = attr_dict[attr][class_name]

        prompt_lst = []
        attr_lst = []
        examples = []
        example_lst = read_first_n_jsonl(args.sample_dir, i, n=args.n_sample, token=30)
        example_index = 0
        # If there is a similar attribute, it means there are similar categories and additional processing is required in the prompt
        if "similar" in attr_dict:
            similar_label = ",".join(attr_dict["similar"][class_name])
        for return_dict in gen_example(attr_dict_cls):
            prompt_tmp = {x: attr_dict_cls[x][return_dict[x]] for x in return_dict}
            attr_lst.append(return_dict)
            if args.dataset == 'amazon':
                #  The review should be relevant to {re.sub('_', ' ', class_name)} and does not discuss {similar_label}.\n \
                prompt_input = f"Write a review for a product about {re.sub('_', ' ', class_name)} on Amazon and follow the requirements below.  \n \
                            The review should be about a product in '{prompt_tmp['product_name']}'\n \
                            The review should avoid starting with common opening phrases like 'I recently purchased ...' or 'I bought ...'.\n"

                # length-words:
                word_length = random.choices([0, 50, 100, 150, 200, 250, 300], weights=[30, 50, 25, 12, 6, 4, 8], k=1)[0]
                prompt_input += f"The review must be between {word_length} to {word_length + 50} words. \n"

                # preset attr
                # attr1: writing style
                prompt_input += prompt_prob(f"The writing style of the review should be {prompt_tmp['style']} \n", 0.5)
                # attr2: logic framework
                prompt_input += prompt_prob(f"The logic framework of the review should be {prompt_tmp['logic']} \n", 0.5)
                # attr3: emotion
                prompt_input += prompt_prob(f"The emotion of the review should be {prompt_tmp['emotion']} \n", 0.5)

                # negative prompt
                negative_lst = []
                for attr_name in attr_name_dict[class_name]:
                    if ":" in attr_name:
                        negative_lst.append(f"'{attr_name.split(':', 1)[0]}'")
                    elif "-" in attr_name:
                        negative_lst.append(f"'{attr_name.split('-', 1)[0]}'")
                    else:
                        negative_lst.append(f"'{attr_name.split(' ')[0]}'")

                prompt_input += f"Do not directly use the words following: {(', '.join(negative_lst)).lower()}.\n"

                # attr_w
                attrs_weights = load_attributes_weights_list(
                    model=args.model_name,
                    dataset=args.dataset,
                    method=args.model_type,
                    c=class_name
                )
                # print(attrs_weights)

                for w in range(len(attrs_weights)):
                    prompt_input += prompt_prob(f"The review should cover the following content: {attr_name_dict[class_name][w]}. \n", attrs_weights[w])

                if example_index < len(example_lst):
                    example = example_lst[example_index]
                else:
                    example = ""

                if random.random() < 1.0 and example != "":
                    prompt_input += f"The review should be like the following example: '{example}' \n"
                    example_index += 1

                # Example:
                # prompt_input += prompt_prob(f"\n The post should be like the following example: '{random_read_jsonl_index(args.sample_dir, i, token=30)}' \n", 1.0)

            elif args.dataset == 'reddit':
                prompt_input = f"Write the content of a web post on Reddit in {re.sub('_', ' ', class_name)} community and follow the requirements below.  \n \
                            The web post does not have no opening remarks or concluding summaries. \n \
                            The post should avoid starting with common opening phrases like 'Anyone else', 'Hey everyone' or 'Just...', etc.\n "

                # length-words
                word_length = random.choices([0, 50, 100, 150, 200, 250, 300], weights=[65, 90, 45, 25, 15, 8, 8], k=1)[0]
                prompt_input += f"The post must be between {word_length} to {word_length + 50} words. \n"

                # preset attr
                # attr1: writing style
                prompt_input += prompt_prob(f"The writing style of the post should be {prompt_tmp['style']} \n", 0.5)
                # attr2: logic framework
                prompt_input += prompt_prob(f"The logic framework of the post should be {prompt_tmp['logic']} \n", 0.5)
                # attr3: emotion
                prompt_input += prompt_prob(f"The emotion of the post should be {prompt_tmp['emotion']} \n", 0.5)

                # similar class_name
                # prompt_input += prompt_prob(f"The post must be irrelevant to: {similar_label}. \n", 1.0)

                # negative prompt
                negative_lst = []
                for attr_name in attr_name_dict[class_name]:
                    if ":" in attr_name:
                        negative_lst.append(f"'{attr_name.split(':', 1)[0]}'")
                    elif "-" in attr_name:
                        negative_lst.append(f"'{attr_name.split('-', 1)[0]}'")
                    else:
                        negative_lst.append(f"'{attr_name.split(' ')[0]}'")

                prompt_input += f"Do not directly use the words following: {(', '.join(negative_lst)).lower()}.\n"

                # attr_w
                attrs_weights = load_attributes_weights_list(
                    model=args.model_name,
                    dataset=args.dataset,
                    method=args.model_type,
                    c=class_name
                )
                # print(attrs_weights)
                for w in range(len(attrs_weights)):
                    prompt_input += prompt_prob(f"The post should cover the following content: {attr_name_dict[class_name][w]}. \n", attrs_weights[w])

                if example_index < len(example_lst):
                    example = example_lst[example_index]
                else:
                    example = ""
                # Example
                if random.random() < 1.0 and example != "":
                    prompt_input += f"The post should be like the following example: '{example}' \n"
                    example_index += 1

                # Example:
                # prompt_input += prompt_prob(f"\n The post should be like the following example: '{random_read_jsonl_index(args.sample_dir, i, token=30)}' \n", 1.0)

            elif args.dataset == 'sst-2':
                prompt_input = "Write a movie review on Rotten Tomatoes Movie Review site and following the requirements below: \n"
                # random_num = random.random()
                # if random_num < 0.1:
                #     if re.sub('_', ' ', class_name) == 'negative':
                #         prompt_input += "The review should be mildly negative. \n"
                #     else:
                #         prompt_input += "The review should be mildly positive. \n"
                # else:
                #     prompt_input += f"The review should be {re.sub('_', ' ', class_name)}. \n"

                # class prompt
                prompt_input += f"The review should be {re.sub('_', ' ', class_name)}. \n"

                # length-words:
                word_length = random.choices([0, 10, 20, 30, 40], weights=[10, 25, 10, 8, 2], k=1)[0]
                prompt_input += f"The review must be between {word_length} to {word_length + 10} words. \n"

                # attrs:
                negative_lst = []
                for attr_name in attr_name_dict[class_name]:
                    if ":" in attr_name:
                        negative_lst.append(f"'{attr_name.split(':', 1)[0]}'")
                    elif "-" in attr_name:
                        negative_lst.append(f"'{attr_name.split('-', 1)[0]}'")
                    else:
                        negative_lst.append(f"'{attr_name.split(' ')[0]}'")

                prompt_input += f"The review should avoid starting with common opening phrases like 'This...', or 'The...'. \n \
                                Do not directly use the words following: {(', '.join(negative_lst)).lower()}. \n "

                # preset attr
                # attr1: writing style
                prompt_input += prompt_prob(f"The writing style of the review should be {prompt_tmp['style']} \n", 0.5)
                # attr2: logic framework
                prompt_input += prompt_prob(f"The logic framework of the review should be {prompt_tmp['logic']} \n", 0.5)
                # attr3: emotion
                prompt_input += prompt_prob(f"The emotion of the review should be {prompt_tmp['emotion']} \n", 0.5)

                attrs_weights = load_attributes_weights_list(
                    model=args.model_name,
                    dataset=args.dataset,
                    method=args.model_type,
                    c=class_name
                )
                # print(attrs_weights)

                for w in range(len(attrs_weights)):
                    prompt_input += prompt_prob(f"The review should cover the following content: {attr_name_dict[class_name][w]}. \n", attrs_weights[w])

                if example_index < len(example_lst):
                    example = example_lst[example_index]
                else:
                    example = ""

                if random.random() < 1 and example != "":
                    prompt_input += f"The review should be like the following example: '{example}' \n"
                    example_index += 1

                # Example
                # prompt_input += prompt_prob(f"\n The review should be like the following example: '{random_read_jsonl_index(args.sample_dir, i, token=30)}' \n", 1.0)

            if attempt == 0 and len(prompt_lst) == 0:
                print(f"First Prompt Input: {prompt_input}")

            prompt_lst.append(
                    [{"role": "user", "content": prompt_input}]
            )
            # prompt_lst达到batch_size时，进行一次尝试
            if len(prompt_lst) == args.batch_size:
                try:
                    attempt += 1
                    return_msg = call_api_async(prompt_lst, args.model_name, args.temperature, args.max_tokens)
                    assert len(return_msg) == len(attr_lst)
                    valid = 0
                    tmp = []
                    for (msg, attr) in zip(return_msg, attr_lst):
                        if "I apologize" in msg or "sorry" in msg or "Sorry" in msg or "an AI language model" in msg or "I cannot perform" in msg:
                            continue
                        else:
                            valid += 1
                            example = {"_id": i, "text": clean_str(msg)}
                            example.update(attr)
                            examples.append(example)
                            tmp.append(example)
                    sent_cnt += valid 
                    prompt_lst = []
                    attr_lst = []
                    print(f"CLass {i}: {class_name}, Attempt: {attempt}, Sent cnt: {sent_cnt}. ")
                    now = datetime.now()
                    formatted_datetime = now.strftime("%Y%m%d%H%M")
                    prefix = f"gen_examples/{args.dataset}/{class_name}/train_attrprompt_{args.model_name}_{args.top_p}_{i}_{attempt}_{formatted_datetime}.jsonl"
                    os.makedirs(f"{args.output_dir}/gen_examples/{args.dataset}/{class_name}", exist_ok=True)
                    f = open(f"{args.output_dir}/{prefix}", 'w')
                    for e in tmp:
                        f.write(json.dumps(e) + "\n")
                    f.close()
                except openai.APIError as e:
                    # 处理OpenAI API特有错误
                    print(f"OpenAI API error: {e}")
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(15)
                    continue
                except Exception as e:
                    # 处理其他所有异常
                    print(f"An unexpected error occurred: {e}")
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue

            if sent_cnt >= args.n_sample or attempt > 200:
                break
       

if __name__ == '__main__':
    main(args)

