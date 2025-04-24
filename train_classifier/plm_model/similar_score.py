from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import argparse
from tqdm import tqdm
import torch


model = SentenceTransformer('all-MiniLM-L6-v2')


def main():
    parser = argparse.ArgumentParser(description='Levenshtein Distance')
    parser.add_argument('input_file', help='JSONL path')
    args = parser.parse_args()
    texts = []
    with open(args.input_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                text = data['text']
                texts.append(text)
            except json.JSONDecodeError:
                print(f"warning: invalid JSON line: {line.strip()}")
            except KeyError:
                print(f"warning: lack text: {line.strip()}")

    if len(texts) < 2:
        print("ERROR: less than 2 lines")
        return

    print("Generating text embeddings...")
    embeddings = []
    for text in tqdm(texts, desc="Embedding texts"):
        embeddings.append(model.encode(text))
    embeddings = np.array(embeddings)
    print("Calculating cosine similarity matrix...")
    similarity_matrix = []
    for i in tqdm(range(len(embeddings)), desc="Calculating similarities"):
        row = []
        for j in range(len(embeddings)):
            row.append(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
        similarity_matrix.append(row)
    similarity_matrix = np.array(similarity_matrix)

    average_similarity = np.mean(similarity_matrix)
    std_similarity = np.std(similarity_matrix)

    lambda_value = 0.5
    diversity_score = 1 - average_similarity + lambda_value * std_similarity

    print(f"Average Similarity: {average_similarity}")
    print(f"Standard Deviation of Similarity: {std_similarity}")
    print(f"Diversity Score: {diversity_score}")


if __name__ == '__main__':
    main()

