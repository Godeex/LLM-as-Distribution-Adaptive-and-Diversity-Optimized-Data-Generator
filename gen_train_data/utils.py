import json 
import re 
from collections import defaultdict
import random


def remove_numbered_lists(text):
    """
    Remove numbered lists (i.e., lines starting with "1. ", "2. ", "3. ", etc.) from the text.

    Args:
    text (str): The text to be processed, which may contain numbered lists.

    Return:
    str: The text after removing numbered lists.
    """
    pattern = r'^\d+\.\s+'
    # Use the re.compile function to compile a regular expression pattern, and set the re.MULTILINE flag
    # The re.MULTILINE flag allows ^ to match the beginning of each line in the text, not just the beginning of the entire string.
    regex = re.compile(pattern, re.MULTILINE)
    
    # Remove numbered lists from the text
    text = regex.sub('', text)
    
    return text


def load_attributes(attr_name="", model="", dataset="", method="", classes=None):
    """
    Based on the provided attribute name (attr_name), model name (model), dataset name (dataset), method name (method), and list of classes (classes),
    return a class-dependent feature dictionary (for certain attribute names) or a generic feature list (for other attribute names).

    Args:
    attr_name (str): The name of the attribute to load.
    model (str): The name of the model, used for constructing file paths.
    dataset (str): The name of the dataset, used for constructing file paths.
    method (str): The name of the method, used for constructing file paths.
    classes (list of str): A list of classes, required only when `attr_name` is a specific value.

    Returns:
    return a dictionary for class-dependent features, and a list for generic features
    dict or list: Depending on the value of `attr_name`, return a dictionary (with keys as classes and values as feature lists for that class) or a feature list.

    """
    return_dict = {}

    if attr_name in ["style", "length", "logic", "emotion"]:
        # If attr_name is one of the above specific values, then load the generic feature list
        lst = []
        with open(f"./datasets/{dataset}/{method}/{model}/{attr_name}/{attr_name}.txt", 'r') as f:
            for lines in f:
                lst.append(lines.strip("\n"))
        return lst
    else:
        assert classes is not None

        for c in classes:
            return_dict[c] = []
            with open(f"./datasets/{dataset}/{method}/{model}/{attr_name}/{c}.jsonl", 'r') as f:
                for lines in f:
                    # Clean up the text, remove lists of numbers, brackets, quotes, etc.
                    clean_text = remove_numbered_lists(lines.lstrip('0123456789.').lstrip('-()*').strip("\"\'\n").strip())
                    # If the cleaned text is not empty, it is added to the feature list of this category.
                    if clean_text != "":
                        return_dict[c].append(clean_text)

    return return_dict


def load_attributes_names(model="", dataset="", method="", classes=None):
    assert classes is not None
    return_dict = {}
    for c in classes:
        return_dict[c] = []
        with open(f"./datasets/{dataset}/{method}/{model}/attribute/{c}.jsonl", 'r') as f:
            for lines in f:
                clean_text = remove_numbered_lists(lines.lstrip('0123456789.').lstrip('-()*').strip("\"\'\n").strip())
                if clean_text != "":
                    return_dict[c].append(clean_text)

    return return_dict


def load_attributes_names_filter(model="", dataset="", method="", classes=None):
    assert classes is not None
    return_dict = {}
    for c in classes:
        return_dict[c] = []
        with open(f"./datasets/{dataset}/{method}/{model}/attr_filter/{c}.jsonl", 'r') as f:
            for lines in f:
                clean_text = remove_numbered_lists(lines.lstrip('0123456789.').lstrip('-()*').strip("\"\'\n").strip())
                if clean_text != "":
                    return_dict[c].append(clean_text)
    return return_dict


def load_attributes_weights(model="", dataset="", method="", classes=None):
    assert classes is not None
    return_dict = {}
    for c in classes:
        merged_dict = {}
        with open(f"./datasets/{dataset}/{method}/{model}/attr_w/{c}.jsonl", 'r') as f:
            for line in f:
                # Parse each line of JSON into a dictionary
                data = json.loads(line.strip())
                # Merge dictionaries, adding values corresponding to the same keys
                for key, value in data.items():
                    if key in merged_dict:
                        merged_dict[key] += value
                    else:
                        merged_dict[key] = value
        return_dict[c] = list(merged_dict.values())

    return return_dict


def load_attributes_weights_list(model="", dataset="", method="", c=None):
    sums = None

    with open(f"./datasets/{dataset}/{method}/{model}/attr_w/{c}.jsonl", 'r', encoding='utf-8') as f:
        l = 0
        for line in f:
            l += 1
            data = json.loads(line.strip())

            if not isinstance(data, list):
                raise ValueError("JSON is not a list")

            if sums is None:
                sums = [0] * len(data)

            for i, value in enumerate(data):
                sums[i] += value

    return [x/l for x in sums]


def prompt_prob(sentence, probability):
    random_num = random.random()
    if random_num < probability:
        return sentence
    else:
        return ""


def prompt_sample(sentences, n):
    selected_sentences = random.sample(sentences, n)
    result = "\n".join(selected_sentences)
    # print(result)
    return result


def read_first_n_jsonl(file_path, index, n=1, token=20):
    selected_jsons = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get('_id') == index:
                words = json_obj.get('text', '').split()[:token]
                text_snippet = ' '.join(words) + "..." if words else ""
                selected_jsons.append(text_snippet)

                if len(selected_jsons) >= n:
                    break

    return selected_jsons


def random_read_jsonl_index(file_path, index, n=1, token=20):
    selected_jsons = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get('_id') == index:
                words = json_obj.get('text').split()[:token]
                text_snippet = ' '.join(words) + "..." if words else ""
                selected_jsons.append(text_snippet)

    if len(selected_jsons) <= n:
        return "\n".join(selected_jsons)

    random_selected = random.sample(selected_jsons, n)

    return "\n".join(random_selected)


def random_read_jsonl_index_list(file_path, index, n=1, token=20):
    selected_jsons = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get('_id') == index:
                words = json_obj.get('text').split()[:token]
                text_snippet = ' '.join(words) + "..." if words else ""
                selected_jsons.append(text_snippet)

    if len(selected_jsons) <= n:
        return selected_jsons

    random_selected = random.sample(selected_jsons, n)

    return random_selected


def random_sample_jsonl_large(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)

    sampled_line_numbers = random.sample(range(total_lines), n)
    sampled_line_numbers.sort()

    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i in sampled_line_numbers:
                data = json.loads(line)
                if 'text' in data:
                    s = data['text']
                    words = s.split()[:100]
                    s_10 = ' '.join(words)+"..."
                    texts.append(s_10)
            if len(texts) >= n:
                break

    return "\n".join(texts)

# if __name__ == "__main__":
