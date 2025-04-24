from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import argparse
import random
from tqdm import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')


def random_read_jsonl_index_list(file_path, index, n=1):
    selected_jsons = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            if json_obj.get('_id') == index:
                text = json_obj.get('text')
                selected_jsons.append(text)

    if len(selected_jsons) <= n:
        return selected_jsons

    random_selected = random.sample(selected_jsons, n)

    return random_selected


def sample_from_jsonl(input_dir, n_samples, class_id):
    texts = random_read_jsonl_index_list(input_dir, index=class_id, n=800)
    texts = list(set(texts))
    if len(texts) < n_samples:
        return texts
    print("Generating text embeddings...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding texts"):
        embeddings.append(model.encode(text))
    embeddings = np.array(embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    average_similarities = np.mean(similarity_matrix, axis=1)

    num_prompts = n_samples
    selected_indices = np.argsort(average_similarities)[:num_prompts]
    selected_samples = [texts[i] for i in selected_indices]

    return selected_samples


def sample_from_jsonl_v2(input_dir, n_samples, class_id):
    texts = random_read_jsonl_index_list(input_dir, index=class_id, n=4000)
    texts = list(set(texts))
    if len(texts) < n_samples:
        return texts
    print("Generating text embeddings...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding texts"):
        embeddings.append(model.encode(text))
    embeddings = np.array(embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    selected_samples = []
    selected_indices = []
    num_prompts = n_samples

    first_index = np.random.choice(len(texts))
    selected_samples.append(texts[first_index])
    selected_indices.append(first_index)

    for _ in tqdm(range(num_prompts - 1), desc="Similar texts"):
        remaining_indices = [i for i in range(len(texts)) if i not in selected_indices]
        avg_similarities = np.mean(similarity_matrix[remaining_indices][:, selected_indices], axis=1)
        next_index = remaining_indices[np.argmin(avg_similarities)]
        selected_samples.append(texts[next_index])
        selected_indices.append(next_index)

    return selected_samples


def sample_from_jsonl_v3(input_dir, n_samples, class_id):
    texts = random_read_jsonl_index_list(input_dir, index=class_id, n=4000)
    texts = list(set(texts))

    print("Generating text embeddings...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding texts"):
        embeddings.append(model.encode(text))
    embeddings = np.array(embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    selected_samples = []
    selected_indices = []

    if len(texts) <= n_samples:
        avg_similarities = np.mean(similarity_matrix, axis=1)
        sorted_indices = np.argsort(avg_similarities)
        selected_samples = [texts[i] for i in sorted_indices[:n_samples]]
        return selected_samples

    first_index = np.random.choice(len(texts))
    selected_samples.append(texts[first_index])
    selected_indices.append(first_index)

    for _ in tqdm(range(n_samples - 1), desc="Selecting diverse samples"):
        remaining_indices = [i for i in range(len(texts)) if i not in selected_indices]
        avg_similarities = np.mean(similarity_matrix[remaining_indices][:, selected_indices], axis=1)
        next_index = remaining_indices[np.argmin(avg_similarities)]
        selected_samples.append(texts[next_index])
        selected_indices.append(next_index)

    return selected_samples


def main():
    parser = argparse.ArgumentParser(description='Levenshtein distance')
    parser.add_argument('--input_file', help='JSONL path')
    parser.add_argument('--output_file', help='JSONL path')
    parser.add_argument('--n_sample', type=int, default=10, help='number of samples')
    parser.add_argument('--class_id', type=int, default=22, help='class id')
    args = parser.parse_args()

    texts_lst = []
    for id in range(args.class_id+1):
        print(f"Class {id} Examples:")
        # texts = sample_from_jsonl(input_dir=args.input_file, n_samples=args.n_sample, class_id=id)
        texts = sample_from_jsonl_v3(input_dir=args.input_file, n_samples=args.n_sample, class_id=id)
        texts_lst.append(texts)

    with open(args.output_file, 'w') as f:
        for i, texts in enumerate(texts_lst):
            for text in texts:
                data = {"_id": i, "text": text}
                f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()

# python test_similar_choice.py --input_file ../../dataset/reddit/train/train-reddit-sort.jsonl --output_file ../../dataset/reddit/train/train-reddit-choice.jsonl --n_sample 100 --class_id 44