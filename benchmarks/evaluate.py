import json

import numpy as np

from benchmarks.dataset import load_balanced_qqp
from scache.core.embedding import encoded


def compute_pair_similarities(pairs: list[tuple[str, str]]) -> np.ndarray:
    q_text1 = [pair[0] for pair in pairs]
    q_text2 = [pair[1] for pair in pairs]
    # shape : (N, 384)
    v1 = encoded(q_text1)
    v2 = encoded(q_text2)
    
    # element wise dot product for each pair
    similarities = np.sum(v1 * v2, axis = 1) # shape : (N,)
    # cosine similarity is simply the dot product as the vectors are L2 normalized
    return np.clip(similarities, -1.0, 1.0)


def sweep_thresholds(duplicate_sims: np.ndarray, non_duplicate_sims: np.ndarray) -> list[dict]:
    results = []
    for tau in np.arange(0.70, 1.00, 0.01): #python range only accepts integer values
        chr_tau = float(np.mean(duplicate_sims >= tau)) # cache hit rate, float coverts numpy.float64 to python float
        whr_tau = float(np.mean(non_duplicate_sims >= tau)) # wrong hit rate, a wrong hit (false acceptance) occurs when two non-duplicate questions score ≥ τ, tricking the cache into returning a wrong answer.
        results.append({
            "tau": round(float(tau), 2), 
            "chr": round(chr_tau, 4),
            "whr": round(whr_tau, 4),
            })

    return results


def find_operational_threshold(sweep_data: list[dict], target_whr: float = 0.01) -> dict:
    safe_candidate = [row for row in sweep_data if row['whr'] <= target_whr]

    if not safe_candidate:
        # return highest threshold if none are under target_whr
        return sweep_data[-1]
    
    return max(safe_candidate, key=lambda x: x["chr"])


def run_evaluation() -> None:
    dup, non_dup = load_balanced_qqp(num_samples=5000)
    dup_sim = compute_pair_similarities(dup)
    non_dup_sim = compute_pair_similarities(non_dup)

    sweep_data = sweep_thresholds(dup_sim, non_dup_sim)

    with open("benchmarks/results.json", "w") as file:
        json.dump(sweep_data, file, indent=4)

    selected_threshold = find_operational_threshold(sweep_data)
    print("=" * 80)
    print("summary table showing tau from 0.70 to 0.99")
    print("=" * 80)
    print("\n")
    print("|     tau     |     chr     |     whr     |")
    for row in sweep_data:
        print(f"|     {row["tau"]}     |     {row["chr"]}     |     {row["whr"]}     |")

    print(f"selected threshold value: {selected_threshold}")


if __name__ == "__main__":
    run_evaluation()
