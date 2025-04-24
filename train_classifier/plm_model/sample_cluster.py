import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_texts_from_jsonl(file_path):
    texts = []
    current_id = None
    current_texts = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            _id = data.get('_id')
            text = data.get('text')
            if _id != current_id:
                if current_texts:
                    texts.append(current_texts)
                current_texts = []
                current_id = _id

            if text:
                current_texts.append(text)

        if current_texts:
            texts.append(current_texts)

    return texts


def text_cluster(batch_size=500, num_clusters=3, texts=[]):
    if len(texts) < num_clusters:
        return texts

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    embeddings = np.array(embeddings).astype('float32')
    print("Embeded")

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
    clusters = kmeans.fit_predict(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    k = 10
    distances, indices = index.search(embeddings, k)

    representative_texts = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) == 0:
            print(f"Cluster {cluster_id} has no samples. Skipping...")
            continue
        cluster_embeddings = embeddings[cluster_indices]
        similarity_matrix = 1 / (1 + distances[cluster_indices])
        average_similarities = np.mean(similarity_matrix, axis=1)

        representative_index = cluster_indices[np.argmax(average_similarities)]
        representative_text = texts[representative_index]
        representative_texts.append(representative_text)

    return representative_texts


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', help='JSONL path')
    parser.add_argument('--output_file', help='JSONL path')
    parser.add_argument('--num_clusters', type=int, default=3, help='number of clusters')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--class_id', type=int, default=0, help='batch size')
    args = parser.parse_args()

    texts = extract_texts_from_jsonl(args.input_file)
    if len(texts) < 2:
        print("ERROR")
        return

    representative_texts = []
    for i, text_lst in enumerate(texts):
        print("Class: ", i)
        if len(text_lst) <= args.num_clusters:
            representative_texts.append(text_lst)
        else:
            representative_texts.append(text_cluster(batch_size=args.batch_size, num_clusters=args.num_clusters, texts=text_lst))
    with open(args.output_file, 'w') as f:
        for i, text_lst in enumerate(representative_texts):
            for text in text_lst:
                data = {"_id": i, "text": text}
                f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()