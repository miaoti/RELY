"""
一条命令重跑论文的全部承重数字（round-23 新增，回应外部审稿"没有真正的干净重生成入口"）。

    python src/reproduce.py            # 按顺序跑一遍；已有产物的实验脚本会断点续跑
    python src/reproduce.py --fresh    # 先把已有产物归档到 results/_archive/，从零重跑
    python src/reproduce.py --quick    # 只跑派生步骤（读现成 CSV，重算 JSON 与图），几秒钟
    python src/reproduce.py --list     # 只打印将要执行的命令，不执行

顺序是有依赖的：
  1) 昂贵实验（写 CSV）：radMLBench 主扫描、RF 稳健性扫描、within-cohort 两个控制实验、
     AIS 队列的精确置换检验与乐观阶梯。
  2) 派生分析（读 CSV，写 JSON 与图）：可靠性预测器、筛查门、RF 对比、基线表。
  3) 图：motivation、method、within_cohort（reliability 与 gate 的图由第 2 步顺带产出）。
  4) 汇总：paper_numbers.py 把论文里出现的每个数字打到一处，便于逐条核对。

第 1 步在普通笔记本上是小时级；--quick 跳过它，只要 results/*.csv 还在就能复现全部
JSON、图和论文数字。
没有 data/ICC.csv（公开发布不带患者表）时，自动跳过三个 AIS 步骤，读缓存产物。
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

# Windows 控制台默认 cp1252，直接 print 中文说明会 UnicodeEncodeError，
# 让一键复现在干净的 Windows 上也能跑完。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 三个步骤要读 AIS 队列的患者级特征表；公开发布不带这张表，缺表时跳过它们，
# 下游步骤改读 results/ 里缓存的产物（ais_forensic.json、permutation_ICC.json、baselines_ICC.csv）。
AIS_TABLE = ROOT / "data" / "ICC.csv"
NEEDS_AIS_TABLE = {"permutation_test.py", "ais_forensic.py", "baselines.py"}

# (脚本, 参数, 是否属于"昂贵实验", 说明)
STEPS = [
    ("run_radmlbench.py", ["--all", "--repeats", "10"], True,
     "50 个公开队列的主扫描 -> results/radmlbench_sweep.csv"),
    ("run_radmlbench_rf.py", ["--repeats", "3"], True,
     "随机森林稳健性扫描（不做单变量预筛）-> results/radmlbench_sweep_rf.csv"),
    ("r_e_selection.py", ["--cohorts", "4", "--reps", "3", "--splits", "50"], True,
     "within-cohort 控制实验 A/B（固定 n 变平衡；固定少数类变 n）"),
    ("r_g_dim_channel.py", ["--cohorts", "4", "--splits", "50", "--seeds", "3"], True,
     "within-cohort 控制实验 C（固定两个类别计数，只变维度；三个选择臂）"),
    # --repeats 20 是必须的：ais_forensic.py 以 repeats=20 现算 honest，并要求置换产物的
    # observed 与它一致，否则报 stale 退出。默认值是 10，会让整条流水线在两步之后失败。
    ("permutation_test.py", ["--exact", "--radiomics-only", "--n-perm", "200",
                             "--repeats", "20"], True,
     "AIS 队列的置换检验，置换的统计量就是 honest_nested_cv 本身"),
    ("ais_forensic.py", [], True,
     "AIS 队列的乐观阶梯与选择乐观 -> results/ais_forensic.json"),
    ("baselines.py", [], True,
     "AIS 队列的诚实特征集基线 -> results/baselines_ICC.csv"),

    ("reliability_predictor.py", [], False,
     "两步分解、留一来源预测、拟合 kappa、计算器 -> JSON + reliability_validation 图"),
    ("screening_gate.py", [], False,
     "建模前可评估性门 -> JSON + screening_gate_roc 图"),
    ("ais_toolkit.py", [], False,
     "把闸门与计算器真正跑在 AIS 队列上 -> results/ais_toolkit.json（§V 的数字来源）"),
    ("analyze_rf_robustness.py", [], False,
     "RF 与 l2-LR 的逐队列对照 -> results/rf_robustness.json"),
    ("fig_motivation.py", [], False, "Fig. 1"),
    ("fig_method.py", [], False, "Fig. 2"),
    ("fig_within_cohort.py", [], False, "Fig. 4"),
    ("paper_numbers.py", [], False, "把论文里的每个数字汇总到一处"),
]


def main():
    ap = argparse.ArgumentParser(description="重跑论文的全部承重数字")
    ap.add_argument("--fresh", action="store_true",
                    help="把已有产物归档到 results/_archive/ 后从零重跑（否则实验脚本断点续跑）")
    ap.add_argument("--quick", action="store_true",
                    help="跳过昂贵实验，只重算 JSON、图和论文数字")
    ap.add_argument("--list", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--stop-on-error", action="store_true", help="任一步失败就停")
    args = ap.parse_args()

    steps = [s for s in STEPS if not (args.quick and s[2])]
    if not AIS_TABLE.exists():
        skipped = [s[0] for s in steps if s[0] in NEEDS_AIS_TABLE]
        steps = [s for s in steps if s[0] not in NEEDS_AIS_TABLE]
        if skipped:
            print("[note] data/ICC.csv not present (the AIS patient table is not part of the public "
                  "release); skipping " + ", ".join(skipped) + "; downstream steps read the cached "
                  "outputs in results/.")
    if args.quick and args.fresh:
        print("[warn] --quick 与 --fresh 同时给出：--fresh 只影响昂贵实验，这里被忽略")

    plan = []
    for script, extra, expensive, note in steps:
        cmd = [PY, str(ROOT / "src" / script), *extra]
        if args.fresh and expensive and script in {
                "run_radmlbench.py", "run_radmlbench_rf.py",
                "r_e_selection.py", "r_g_dim_channel.py"}:
            cmd.append("--fresh")
        plan.append((cmd, note))

    for i, (cmd, note) in enumerate(plan, 1):
        print("[%d/%d] %s" % (i, len(plan), note))
        print("      " + " ".join(Path(c).name if c.endswith(".py") else c for c in cmd))
    if args.list:
        return 0

    t0, failed = time.time(), []
    for i, (cmd, note) in enumerate(plan, 1):
        print("\n=== [%d/%d] %s" % (i, len(plan), note), flush=True)
        # 子进程也要 utf-8：Windows 控制台默认 cp1252，脚本里的中文说明会让子进程
        # 直接 UnicodeEncodeError 退出（paper_numbers.py 就是这么挂的）。
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        r = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            failed.append((cmd[1], r.returncode))
            print("!!! 失败：%s (exit %d)" % (Path(cmd[1]).name, r.returncode))
            if args.stop_on_error:
                break
    print("\n用时 %.1f 分钟" % ((time.time() - t0) / 60))
    if failed:
        print("失败的步骤：" + ", ".join("%s(%d)" % (Path(a).name, b) for a, b in failed))
        return 1
    print("全部完成。论文数字见上一步 paper_numbers.py 的输出；"
          "图在 figures/，复制到 paper/figures/ 后重编译 paper/main.tex。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
