"""
ISLES'24 → PyRadiomics 特征表（与 data/ICC.csv 同构，供 src/honest_eval.py --data 直接复用）。

⚠️ 重要：本脚本【未在本机验证】——本机无数据、无 PyRadiomics 环境。它是按 IBSI 与执行清单
   阶段 3.5 要求写的【可运行模板】。拿到数据后，请先 `--dry-run` 核对 (影像,掩膜) 配对与临床表，
   再正式提取；并按你看到的真实 BIDS 目录调 --img-glob/--mask-glob/--id-regex/--clinical 等。

⚠️ 环境：PyRadiomics 不支持 numpy>=2 / Python 3.12。必须单独建兼容环境，例如：
     conda create -n radiomics python=3.10 -y
     conda activate radiomics
     pip install "numpy<2" SimpleITK pyradiomics nibabel pandas
   （当前项目主环境是 py3.12+numpy2，跑不了 PyRadiomics；提取在这个副环境里做即可。）

结局约定（与 ICC.csv 对齐）：列名 `Categories`，**1=好预后(mRS 0–2)，0=差预后(mRS 3–6)**；
   honest_eval.py 取 阳性=poor=(Categories==0)。

数据获取：Zenodo record 16748089（ISLES'24，CC-BY-NC-SA 非商用），单个 train.7z ≈ 99 GB，
   需自行下载并 7z 解压，再把解压根目录传给 --root。

用法：
   python src/extract_isles.py --root /path/to/isles24 --clinical /path/clinical.csv --dry-run
   python src/extract_isles.py --root /path/to/isles24 --clinical /path/clinical.csv \
          --img-glob "**/*dwi*.nii.gz" --mask-glob "**/*lesion-msk*.nii.gz" \
          --id-regex "(sub-[A-Za-z0-9]+)" --id-col subject --mrs-col mrs_90d \
          --out data/isles24/features.csv
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# IBSI 合规的 MRI 提取设置（可复现是本文卖点，全程记录到 results/）
EXTRACTION_SETTINGS = {
    "normalize": True,            # z-score 强度归一化（MRI 必须）
    "normalizeScale": 100,
    "binWidth": 5,                # 固定 bin width（IBSI 推荐固定，而非固定 bin 数）
    "resampledPixelSpacing": [1, 1, 1],   # 各向同性体素重采样
    "interpolator": "sitkBSpline",
    "label": 1,                   # 掩膜中病灶标签值（按数据调整）
    "force2D": False,
}
IMAGE_TYPES = ["Original", "Wavelet"]     # 对齐 ICC 的 原始 + 小波子带


def discover_pairs(root: Path, img_glob: str, mask_glob: str, id_regex: str):
    """按 glob 找影像与掩膜，按 id_regex 抽取 subject id 配对。"""
    pat = re.compile(id_regex)
    def index(glob):
        out = {}
        for p in root.glob(glob):
            m = pat.search(str(p))
            if m:
                out.setdefault(m.group(1), p)
        return out
    imgs, masks = index(img_glob), index(mask_glob)
    ids = sorted(set(imgs) & set(masks))
    missing = sorted((set(imgs) | set(masks)) - set(ids))
    return [(i, imgs[i], masks[i]) for i in ids], missing


def align_mask(image, mask, sitk):
    """几何不一致时把掩膜最近邻重采样到影像空间（否则 PyRadiomics 报错或提错）。"""
    same = (image.GetSize() == mask.GetSize()
            and image.GetSpacing() == mask.GetSpacing()
            and image.GetDirection() == mask.GetDirection()
            and image.GetOrigin() == mask.GetOrigin())
    if same:
        return mask, False
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(image)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(mask), True


def load_outcome(clinical_csv: Path, id_col: str, mrs_col: str):
    """读临床表，mRS → 二值 Categories（1=好 mRS<=2 / 0=差 mRS>=3）。"""
    import pandas as pd
    df = pd.read_csv(clinical_csv)
    for c in (id_col, mrs_col):
        if c not in df.columns:
            sys.exit(f"[error] clinical csv 缺列 '{c}'；实际列：{list(df.columns)}")
    mrs = pd.to_numeric(df[mrs_col], errors="coerce")
    df["Categories"] = (mrs <= 2).astype("Int64")        # 1=good, 0=poor
    return df.set_index(id_col)["Categories"].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ISLES'24 解压根目录")
    ap.add_argument("--clinical", required=True, help="临床表 csv（含 subject id 与 mRS）")
    ap.add_argument("--img-glob", default="**/*dwi*.nii.gz")
    ap.add_argument("--mask-glob", default="**/*lesion-msk*.nii.gz")
    ap.add_argument("--id-regex", default=r"(sub-[A-Za-z0-9]+)")
    ap.add_argument("--id-col", default="subject")
    ap.add_argument("--mrs-col", default="mrs_90d")
    ap.add_argument("--out", default=str(ROOT / "data" / "isles24" / "features.csv"))
    ap.add_argument("--dry-run", action="store_true", help="只列配对与结局覆盖，不提取")
    args = ap.parse_args()

    root = Path(args.root)
    pairs, missing = discover_pairs(root, args.img_glob, args.mask_glob, args.id_regex)
    print(f"[discover] paired={len(pairs)}  unpaired(缺影像或掩膜)={len(missing)}")
    if missing[:10]:
        print("  unpaired sample:", missing[:10])
    outcome = load_outcome(Path(args.clinical), args.id_col, args.mrs_col)
    have_y = [i for i, _, _ in pairs if outcome.get(i) is not None]
    print(f"[outcome] 有 mRS 标签的配对样本：{len(have_y)}/{len(pairs)}")

    if args.dry_run:
        for i, im, mk in pairs[:8]:
            print(f"  {i}  y={outcome.get(i)}  img={im.name}  mask={mk.name}")
        print("[dry-run] 核对无误后去掉 --dry-run 正式提取。")
        return

    # 仅在正式提取时才 import 重依赖，便于 --dry-run 在主环境也能跑
    import SimpleITK as sitk
    import pandas as pd
    from radiomics import featureextractor

    ext = featureextractor.RadiomicsFeatureExtractor(**EXTRACTION_SETTINGS)
    ext.disableAllImageTypes()
    for t in IMAGE_TYPES:
        ext.enableImageTypeByName(t)
    ext.enableAllFeatures()

    rows = []
    for n, (sid, img_p, mask_p) in enumerate(pairs, 1):
        y = outcome.get(sid)
        if y is None:
            continue
        try:
            img = sitk.ReadImage(str(img_p))
            msk = sitk.ReadImage(str(mask_p))
            msk, resampled = align_mask(img, msk, sitk)
            feats = ext.execute(img, msk)
            row = {"SubjectID": sid, "Categories": int(y)}
            row.update({k: float(v) for k, v in feats.items()
                        if not k.startswith("diagnostics_")})
            row["_mask_resampled"] = int(resampled)
            rows.append(row)
            print(f"  [{n}/{len(pairs)}] {sid} ok ({len(row)-3} feats)"
                  + ("  [mask resampled]" if resampled else ""))
        except Exception as e:
            print(f"  [{n}/{len(pairs)}] {sid} FAILED: {type(e).__name__}: {e}")

    if not rows:
        sys.exit("[error] 没有成功提取任何样本——请用 --dry-run 核对配对/标签/路径。")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    cfg = {"settings": EXTRACTION_SETTINGS, "image_types": IMAGE_TYPES,
           "n_samples": len(df), "n_features": df.shape[1] - 3,
           "outcome_rule": "Categories: 1=good(mRS<=2), 0=poor(mRS>=3)",
           "source": "Zenodo 16748089 ISLES'24 (CC-BY-NC-SA)"}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "isles_extraction_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}  ({len(df)} samples × {df.shape[1]-3} features)")
    print("next: python src/honest_eval.py --data", out)


if __name__ == "__main__":
    main()
