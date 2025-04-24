import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os
from tqdm import trange, tqdm
import re
import time
import json
from utils import random_read_jsonl_index_list
import random


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--n_attribute", default=10, type=int)
parser.add_argument("--dataset", default='amazon', type=str)
parser.add_argument("--attribute", default='topic', type=str)

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='dado', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=2048, type=int)
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")
parser.add_argument("--n_sample", default=10, type=int, help="the number of examples generated for each class")
parser.add_argument("--sample_dir", default='./dataset/amazon/train/train-amazon-cluster.jsonl', type=str, help="the sample jsonl file")

args = parser.parse_args()

api_key = ' '
base_url = ' '


if args.dataset in ['amazon']:
    args.domain = 'product reviews'
    args.data_source = "Amazon"
    args.attributes = ["product_name", "similar", "length", "style", "logic", "emotion"]
    args.example_attr = "product_name"
elif args.dataset in ['sst-2']:
    args.domain = 'movie reviews'
    args.data_source = "Rotten Tomatoes movie review site"
    args.attributes = ["length", "style", "logic", "emotion"]
    args.example_attr = "genre"
elif args.dataset in ['reddit']:
    args.domain = 'web post'
    args.data_source = args.dataset
    args.attributes = ["similar", "length", "style", "logic", "emotion"]
    args.example_attr = "topic"
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
    print("===================================")
    print(f"Calling APIs, {len(msg_lst)} requests in total.")

    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(
        dispatch_openai_requests_async(
            messages_list=msg_lst,
            model=model,
            temperature=1.0,
            max_tokens=400,
            top_p=1.0,
        )
    )

    # Extract the generated text from the responses
    ans = [x.choices[0].message.content for x in response]
    # print(ans)

    print(f"API returned {len(ans)} responses in total.")
    print("===================================")
    return ans

# Example usage:
# msg_lst = [...]  # Your list of message lists
# model = "gpt-3.5-turbo"  # Or "gpt-4" or any other supported model
# responses = call_api_async(msg_lst, model)
# print(responses)


def remove_empty_lines(text):
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return "\n".join(non_empty_lines)


def main(args):
    with open(f"./datasets/{args.dataset}/label.txt", 'r') as f:
        label_names = [x.lower().replace(" ", "_").strip('\n') for x in f.readlines()]
    print(label_names)

    args.prompt = [
        re.sub("_", " ", f"List {args.n_sample} diverse attributes of {label_names[i]} {args.domain} from {args.data_source} like '{args.example_attr}'. \n \
                        The attributes should not be related to {', '.join(args.attributes)}. \n \
                        The attributes should not be related to each other. \n \
                        The attributes should not be related to non textual content such as images. \n \
                        The attributes should not contain attribute values. \n \
                        Write each attribute and its explanation in one line no more than 10 words use '1. 2. 3.'. \n"
        ) for i in range(len(label_names))
    ]

    # for i in range(len(args.prompt)):
    #     texts = random_read_jsonl_index_list(args.sample_dir, index=i, n=5, token=50)
    #     text_examples = "\n".join(texts)
    #     args.prompt[i] += f"The attributes should be extracted from the provided examples below: {text_examples}"

    print("Prompt 0:", args.prompt[0])

    msg_lst = [[{"role": "user", "content": p}] for p in args.prompt]
    msg_batches = [msg_lst[i:i + 3] for i in range(0, len(msg_lst), 3)]

    success = 0
    return_msg = []
    for batch in tqdm(range(len(msg_batches)), desc="API", unit="times"):
        while success == 0:
            try:
                return_msg += call_api_async(msg_batches[batch], args.model_name)
                success = 1
            except openai.APIError as e:
                print("Not succeed. Try again.")
                continue
        success = 0

    assert len(return_msg) == len(label_names)
    for msg, x in zip(return_msg, label_names):
        class_name = re.sub(" ", "_", x)
        prefix = f"{args.dataset}/{args.model_name}"
        os.makedirs(f"{args.output_dir}/{prefix}/", exist_ok=True)
        f = open(f"{args.output_dir}/{prefix}/{class_name}.jsonl", 'w')
        f.write(remove_empty_lines(msg) + '\n')


if __name__ == '__main__':
    main(args)

