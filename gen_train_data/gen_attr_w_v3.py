import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os
from tqdm import trange, tqdm
import re
import time
import json
from utils import load_attributes, load_attributes_names, load_attributes_names_filter, \
    random_read_jsonl_index, random_read_jsonl_index_list, prompt_prob
import numpy as np


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def split_list(lst, k):
    """Split a list into chunks of length k"""
    return [lst[i:i+k] for i in range(0, len(lst), k)]


parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=0.7, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--batch_size", default=20, type=int, help="")
parser.add_argument("--dataset", default='amazon', type=str, help="which dataset to use")
parser.add_argument("--attribute", default='topic', type=str, help="which attribute to use")
parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='dado', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=400, type=int, help="maximum tokens")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")
parser.add_argument("--sample_dir", default='./dataset/reddit/train/train-reddit-choice.jsonl', type=str, help="the folder for saving the example text")
parser.add_argument("--n_sample", default=3, type=int, help="the number of examples generated for each class")


args = parser.parse_args()

api_key = ' '
base_url = ' '

args.api_key = api_key
args.base_url = base_url

if args.dataset in ['amazon']:
    args.domain = 'Product Review'
    args.data_source = "Amazon"
elif args.dataset in ['reddit']:
    args.domain = 'web post'
    args.data_source = args.dataset
elif args.dataset in ['sst-2']:
    args.domain = 'movie review'
    args.data_source = 'Rotten Tomatoes Movie Review site'
else:
    raise NotImplementedError


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


def call_api_async(msg_lst, model):
    # print("===================================")
    # print(f"Calling APIs, {len(msg_lst)} requests in total.")

    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        dispatch_openai_requests_async(
            messages_list=msg_lst,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
    )

    # Extract the generated text from the responses
    ans = [x.choices[0].message.content for x in response]
    # print(ans)

    # print(f"API returned {len(ans)} responses in total.")
    # print("===================================")
    return ans


def main(args):
    with open(f"./datasets/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().replace(" ", "_").strip('\n') for x in f.readlines()]
    print(label_names)

    model = args.model_name
    prompt = args.prompt
    openai.api_key = args.api_key
    return_dict = {}

    # 返回dict -> class: [attr_list]
    attr_name_dict = load_attributes_names(
        model=args.model_name,
        dataset=args.dataset,
        method=args.model_type,
        classes=label_names
    )

    for i, c in enumerate(label_names):
        print(f"Class {c} following...")
        dict_lst = []
        num_lst = []
        examples = random_read_jsonl_index_list(args.sample_dir, index=i, n=args.n_sample, token=200)
        print("Example:", len(examples))
        for s in tqdm(range(len(examples))):
            # example = random_read_jsonl_index(args.sample_dir, i, token=200)
            args.prompt = [
                re.sub("_", " ", f"Consider a {c} {args.domain} from {args.data_source}: {examples[s]}. \
                Is it relevant to the following attribute: {x}? \
                Return 1 for yes and 0 for no. You should return only one number. If you are not sure, also return 0. ") for x in attr_name_dict[c]
            ]
            msg_lst = [
                [{"role": "user", "content": p}] for p in args.prompt
            ]

            succeed = 0
            while succeed == 0:
                try:
                    return_msg = call_api_async(msg_lst, args.model_name)
                    # print(attributes[c], return_msg)
                    # assert len(return_msg) == len(prompt) and len(prompt) == len(attributes[c])
                    prefix = f"{args.dataset}/{args.model_name}/attr_w"
                    os.makedirs(f"{args.output_dir}/{prefix}/", exist_ok=True)
                    f = open(f"{args.output_dir}/{prefix}/{c}.jsonl", 'w')
                    res_list = []
                    for msg in return_msg:
                        msg = msg.lower()
                        if 'yes' in msg or '1' in msg:
                            res_list.append(1)
                        else:
                            res_list.append(0)
                    dict_lst.append(dict(zip(attr_name_dict[c], res_list)))
                    num_lst.append(res_list)
                    succeed = 1
                    # exit()
                except openai.APIError as e:
                    # 处理OpenAI API特有错误
                    print(f"OpenAI API error: {e}")
                    time.sleep(15)
                except Exception as e:
                    # 处理其他所有异常
                    print(f"An unexpected error occurred: {e}")
                    time.sleep(5)

        # for data in dict_lst:
        #     f.write(json.dumps(data) + '\n')
        for data in num_lst:
            f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main(args)

