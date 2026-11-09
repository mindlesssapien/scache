from datasets import load_dataset


def load_balanced_qqp(
        num_samples: int = 10000, split: str = "validation", seed: int = 42
        ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    dataset = load_dataset("SetFit/qqp", split=split)
    pos = dataset.filter(lambda x: x["label"] == 1).shuffle(seed=seed) # duplicates
    neg = dataset.filter(lambda x: x["label"] == 0).shuffle(seed=seed) # non duplicates
    half = min(num_samples//2, len(pos), len(neg)) # so avoid Index error as the validation slipt as a total of only 14900 duplicate pairs
    
    pos_subset = pos.select(range(half))
    neg_subset = neg.select(range(half))
    duplicates = list(zip(pos_subset["text1"], pos_subset["text2"]))
    non_duplicates = list(zip(neg_subset["text1"], neg_subset["text2"]))


    return (duplicates, non_duplicates)


if __name__ == "__main__":
    d, nd = load_balanced_qqp(num_samples=1000)
    print(f"length of duplicates: {len(d)}, length of non duplicates: {len(nd)}")
    print(f"sample duplicate pair: {d[0]}")
    print(f"sample non duplicate pair: {nd[0]}")

