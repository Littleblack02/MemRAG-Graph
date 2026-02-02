import argparse
import json
import time
import math


def read_jsonl(file_path):
    data = list()
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f'[{time.asctime()}] Read {len(data)} from {file_path}')
    return data


def dcg(rels):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))


def compute_metrics(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values=[2, 5, 10]):
    """
    Pure-Python metrics: precision, recall, MAP, NDCG at K.
    """
    metrics = {f'precision_at_{k}': 0.0 for k in k_values}
    metrics.update({f'recall_at_{k}': 0.0 for k in k_values})
    metrics.update({f'map_at_{k}': 0.0 for k in k_values})
    metrics.update({f'ndcg_at_{k}': 0.0 for k in k_values})

    num_q = len(qrels) if qrels else 1
    for qid, rel_docs in qrels.items():
        pred = results.get(qid, {})
        ranked = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        rel_set = set(rel_docs.keys())

        for k in k_values:
            topk = ranked[:k]
            hits = [1 if doc_id in rel_set else 0 for doc_id, _ in topk]
            # precision / recall
            metrics[f'precision_at_{k}'] += sum(hits) / (k if k else 1)
            metrics[f'recall_at_{k}'] += sum(hits) / (len(rel_set) if rel_set else 1)
            # MAP@k
            prec_prefix = []
            hit_count = 0
            for idx, h in enumerate(hits, start=1):
                if h:
                    hit_count += 1
                    prec_prefix.append(hit_count / idx)
            if len(rel_set) > 0:
                metrics[f'map_at_{k}'] += (sum(prec_prefix) / len(rel_set)) if prec_prefix else 0.0
            # NDCG@k
            ideal = [1] * min(len(rel_set), k)
            metrics[f'ndcg_at_{k}'] += dcg(hits) / (dcg(ideal) if ideal else 1)

    for key in metrics.keys():
        metrics[key] /= num_q
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval results.')
    parser.add_argument('--gold', '-g', type=str, required=True, help='Path to the gold file.')
    parser.add_argument('--pred', '-p', type=str, required=True, help='Path to the predicted file.')
    args = parser.parse_args()
    gold_data = read_jsonl(args.gold)
    pred_data = read_jsonl(args.pred)
    assert len(gold_data) == len(pred_data), "Gold and pred files must have the same number of entries."

    qrels = {i['id']: {d: 1 for d in i['supporting_ids']} for i in gold_data}
    results = {i['id']: {d: s for d, s in i['retrieved_docs']} for i in pred_data}
    metrics = compute_metrics(qrels, results)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    main()

"""
pip install pytrec_eval pandas
"""
