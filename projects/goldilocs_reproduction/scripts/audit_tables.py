"""Audit published table arithmetic. Does NOT run or evaluate GOLDILOCS."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "audit/paper_reported_metrics.json").read_text(encoding="utf-8"))
metrics = data["metric_order"]


def compare(rows, main):
    return {
        name: {
            "equal_scene_mean": round(mean(row[index] for row in rows), 6),
            "paper_main_table": main[index],
            "main_minus_equal_scene_mean_pp": round(main[index] - mean(row[index] for row in rows), 6),
        }
        for index, name in enumerate(metrics)
    }


result = {
    "kind": "published_table_arithmetic_only",
    "evaluated_model_samples": 0,
    "assumption": "Equal scene weighting; authors' exact aggregation needs clarification.",
    "3DGS-CD": {
        method: compare([scene[method] for scene in data["table7"].values()], data["table4"]["3DGS-CD"][method])
        for method in ("3DGS-CD", "GOLDILOCS")
    },
    "NeRFCD": compare(list(data["table6_goldilocs"].values()), data["table4"]["NeRFCD"]["GOLDILOCS"]),
}
result["3DGS-CD"]["goldilocs_minus_baseline_f1_pp"] = round(
    result["3DGS-CD"]["GOLDILOCS"]["f1"]["equal_scene_mean"]
    - result["3DGS-CD"]["3DGS-CD"]["f1"]["equal_scene_mean"], 6
)
(ROOT / "audit/table_consistency.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
paper = ROOT / "papers/GOLDILOCS_ICLR2026.pdf"
manifest = {
    "paper_sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
    "mast3r_commit": "f5209afc300cec36239a7ac992263f36847bbba0",
    "dust3r_submodule_commit": "3cc8c88c413bb9e34c41db0e0eef99c2ee010b12",
    "croco_submodule_commit": "d7de0705845239092414480bd829228723bf20de",
    "goldilocs_repository_tree_sha": "29d415e2ee066f929e1fa43a6b78396a7569a4ea",
    "note": "Repository tree SHA is not a commit SHA. Dependencies and checkpoints are not fully installed."
}
(ROOT / "audit/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
