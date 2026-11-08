import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scache.cache.semantic_cache import SemanticCache
from scache.core.embedding import encoded


def run_verification():
    print("=" * 80)
    print(" Phase 1 Verification: Core Vector Engine & Local Semantic Matching")
    print("=" * 80)

    # Initialize cache with baseline threshold 0.85
    cache = SemanticCache(threshold=0.85)

    # 1. Populate canonical seed entries
    seed_data = [
        (
            "How do I reverse a linked list in Python?",
            "def reverse_list(head): prev = None; ... return prev",
        ),
        (
            "What is the capital of France?",
            "The capital of France is Paris.",
        ),
        (
            "How does backpropagation work in neural networks?",
            "Backpropagation computes gradients of the loss with respect to weights using the chain rule.",
        ),
    ]

    print("\n[+] Populating cache with canonical entries:")
    for prompt, response in seed_data:
        cache.update(prompt, response)
        print(f"  • Seeded: '{prompt}'")

    cache.check_health()
    print(f"[+] Active Threshold: {cache.threshold}\n")

    # 2. Define test scenarios: (Category, Query, Expected Hit?)
    test_cases = [
        (
            "Exact Match",
            "How do I reverse a linked list in Python?",
            True,
        ),
        (
            "Semantic Equivalent (Paraphrase)",
            "Python code to invert a linked list",
            True,
        ),
        (
            "Semantic Equivalent (Rephrased Question)",
            "Which city serves as the capital of France?",
            True,
        ),
        (
            "Structural Trap (Swapped Entity: Linked List -> Binary Tree)",
            "How do I reverse a binary tree in Python?",
            False,
        ),
        (
            "Structural Trap (Swapped Entity: France -> Germany)",
            "What is the capital of Germany?",
            False,
        ),
        (
            "Completely Unrelated",
            "How do you bake sourdough bread from scratch?",
            False,
        ),
    ]

    print("-" * 80)
    print(
        f"{'Category':<32} | {'Raw Score':<10} | {'Status':<6} | {'Expected':<8} | {'Result'}"
    )
    print("-" * 80)

    all_passed = True

    for category, query, expected_hit in test_cases:
        # Check lookup via cache facade
        lookup_res = cache.lookup(query)
        is_hit = lookup_res is not None

        # Inspect raw score directly from storage for diagnostics
        query_vec = encoded(query)
        raw_res = cache.store.search(query_vec)
        raw_score = raw_res[0] if raw_res else 0.0

        status_str = "HIT" if is_hit else "MISS"
        expected_str = "HIT" if expected_hit else "MISS"

        passed = (is_hit == expected_hit)
        if not passed:
            all_passed = False

        result_tag = "[PASS]" if passed else "[WARN/MISMATCH]"

        print(
            f"{category:<32} | {raw_score:<10.4f} | {status_str:<6} | {expected_str:<8} | {result_tag}"
        )
        print(f"  Query: \"{query}\"")
        if raw_res:
            matched_prompt = raw_res[1].get("prompt", "")
            print(f"  Matched Against: \"{matched_prompt}\"")
        if lookup_res:
            print(f"  Returned Response: {lookup_res[0][:60]}...")
        print()

    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All test cases behaved as expected with threshold = 0.85!")
    else:
        print("[NOTE] Some boundary cases differed from expectations.")
        print("This highlights why Phase 2 (statistical threshold tuning on QQP) is essential!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
