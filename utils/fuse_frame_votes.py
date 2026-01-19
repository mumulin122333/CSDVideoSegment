#!/usr/bin/env python3
import sys, csv, collections, math, os

if len(sys.argv) < 6:
    print("Usage: fuse_vote.py <final_csv> <thresh> <w_hi> <w_lo> <csv1> [<csv2> ...]")
    sys.exit(1)

final_csv = sys.argv[1]
thresh = float(sys.argv[2])
w_hi = float(sys.argv[3])
w_lo = float(sys.argv[4])
csv_paths = sys.argv[5:]

preds = collections.defaultdict(list)   # filename -> [(label, prob, weight, model_idx)]
labels_seen = set()
files_seen_per_model = []               # for duplicate check

# 讀每個模型的 CSV
for mi, path in enumerate(csv_paths):
    if not os.path.isfile(path):
        print(f"⚠️  {path} 不存在，跳過")
        continue
    files_seen = set()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        need = {'filename','top1_label','top1_prob'}
        if not need.issubset(reader.fieldnames):
            raise ValueError(f"{path} 缺少欄位 {need}，實際欄位: {reader.fieldnames}")
        for row in reader:
            fn = row['filename']
            if fn in files_seen:
                # 同一模型 CSV 內有重複行，跳過重複
                continue
            try:
                label = row['top1_label']
                prob = float(row['top1_prob'])
            except Exception:
                continue
            w = w_hi if prob >= thresh else w_lo
            preds[fn].append((label, prob, w, mi))
            labels_seen.add(label)
            files_seen.add(fn)
    files_seen_per_model.append((mi, len(files_seen)))

# 融合投票
rows = []
for fn, lst in preds.items():
    # label -> [sum_weight, sum_weighted_prob]
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for label, prob, w, _ in lst:
        agg[label][0] += w
        agg[label][1] += w * prob

    # 先比加權票數，再比加權機率和，再比 label 字典序（確保可重現）
    best_label, (best_w, best_wp) = max(
        agg.items(),
        key=lambda kv: (kv[1][0], kv[1][1], str(kv[0]))
    )
    total_w = sum(v[0] for v in agg.values())
    avg_prob = best_wp / (best_w if best_w > 0 else 1e-9)

    # 也把每類統計攤開（方便除錯，可視需要註解掉）
    detail_votes = {f"votes_{k}": f"{v[0]:.3f}" for k, v in agg.items()}
    detail_wprob = {f"wprob_{k}": f"{v[1]:.3f}" for k, v in agg.items()}

    rows.append({
        'filename': fn,
        'final_label': best_label,
        'final_weighted_votes': f"{best_w:.3f}/{total_w:.3f}",
        'final_weighted_avg_prob': f"{avg_prob:.4f}",
        'models_considered': len(lst),
        **detail_votes,
        **detail_wprob,
    })

rows.sort(key=lambda r: r['filename'])

# 動態欄位（涵蓋所有見過的 label 的細節欄）
vote_cols  = [f"votes_{l}" for l in sorted(labels_seen, key=str)]
wprob_cols = [f"wprob_{l}" for l in sorted(labels_seen, key=str)]
fieldnames = ['filename','final_label','final_weighted_votes','final_weighted_avg_prob','models_considered'] + vote_cols + wprob_cols

os.makedirs(os.path.dirname(final_csv), exist_ok=True)
with open(final_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        # 確保缺少的欄位也有 key（避免 KeyError）
        for k in vote_cols + wprob_cols:
            r.setdefault(k, "")
        writer.writerow(r)

print(f"✅ fused CSV saved to {final_csv}")
