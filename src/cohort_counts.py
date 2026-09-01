"""
队列的类别计数与 held-out 测试集计数 —— 全项目唯一真源（round-23 新增）。

为什么需要这个模块（外部审稿 round-23 指出的两个硬伤，均已复现）：

① **正类方向错**：`run_radmlbench.py` 只存了 `minority = min(n_pos, n_neg)`，而下游把
   `minority` 直接当成 Hanley-McNeil 公式里的 n_+。但 AUC 是 `roc_auc_score(y, ...)` 算的，
   正类是 `Target==1`。实测 50 个 radMLBench 队列里 **27 个** 的 Target=1 是多数类，
   对这些队列 n_+ 与 n_- 被整体交换了。HM 公式对两者【不对称】
   （q1=a/(2-a) ≠ q2=2a²/(1+a)），所以交换会改变 SE。

② **测试集计数用了 round(0.3*count)**：真正的 `train_test_split(test_size=0.30, stratify=y)`
   按分层分配，实测 **28/50** 个队列至少有一类与四舍五入值差 1。分层划分的类别计数只由
   (n_pos, n_neg, test_size) 决定、与 random_state 无关，所以可以精确算出来，
   不需要重跑 sweep。

用法：
    from cohort_counts import counts_table
    t = counts_table(list_of_dataset_names)   # -> DataFrame，含 n_pos/n_neg/te_pos/te_neg
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cohort_counts.csv"
TEST_SIZE = 0.30


def exact_test_counts(n_pos: int, n_neg: int, test_size: float = TEST_SIZE):
    """sklearn 分层划分实际给出的测试集类别计数。

    分层划分的每类计数只依赖 (n_pos, n_neg, test_size)，与 random_state 无关，
    所以用任意种子跑一次即可，结果是精确的而非估计。"""
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    _, te = train_test_split(np.arange(len(y)), test_size=test_size,
                             stratify=y, random_state=0)
    return int(y[te].sum()), int((1 - y[te]).sum())


def counts_table(datasets, use_cache: bool = True) -> pd.DataFrame:
    """每个队列的真实 n_pos/n_neg（正类 = Target==1，与 roc_auc_score 一致）
    以及精确的 held-out 测试集计数。结果缓存到 results/cohort_counts.csv。"""
    if use_cache and CACHE.exists():
        t = pd.read_csv(CACHE)
        if set(datasets) <= set(t["dataset"]):
            return t.set_index("dataset").loc[list(datasets)].reset_index()

    import radMLBench as rb
    rows = []
    for name in datasets:
        df = rb.loadData(name)
        y = df["Target"].astype(int).values
        n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
        te_pos, te_neg = exact_test_counts(n_pos, n_neg)
        rows.append({"dataset": name, "n": n_pos + n_neg,
                     "n_pos": n_pos, "n_neg": n_neg,
                     "minority": min(n_pos, n_neg),
                     "pos_is_majority": bool(n_pos > n_neg),
                     "te_pos": te_pos, "te_neg": te_neg})
    t = pd.DataFrame(rows)
    CACHE.parent.mkdir(exist_ok=True)
    t.to_csv(CACHE, index=False)
    return t


def main():
    d = pd.read_csv(ROOT / "results" / "radmlbench_sweep.csv")
    t = counts_table(list(d["dataset"]), use_cache=False)
    swapped = int((t["minority"] != t["n_pos"]).sum())
    rounded_pos = np.maximum(1, np.round(0.30 * t["n_pos"]).astype(int))
    rounded_neg = np.maximum(1, np.round(0.30 * t["n_neg"]).astype(int))
    off = int(((rounded_pos != t["te_pos"]) | (rounded_neg != t["te_neg"])).sum())
    print(f"cohorts: {len(t)}")
    print(f"  Target=1 is the majority class in {int(t['pos_is_majority'].sum())} cohorts")
    print(f"  n_pos != minority (old code would swap) in {swapped} cohorts")
    print(f"  round(0.3*count) != exact stratified test counts in {off} cohorts")
    print(f"saved -> {CACHE}")


if __name__ == "__main__":
    main()
