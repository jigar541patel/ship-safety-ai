#!/usr/bin/env python3
"""
Painting Mantra Safety — YOLO Training Script
=============================================
Run this in Google Colab (free T4 GPU recommended).

Phase 1: Train on 8 Roboflow datasets → 11 classes
Phase 2: Add client photos → 16 classes (using Phase 1 weights)

"""

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not ROBOFLOW_API_KEY:
    raise ValueError("❌ Please set ROBOFLOW_API_KEY as environment variable")
    
YOUR_WORKSPACE   = "YOUR_WORKSPACE_SLUG" # your Roboflow workspace slug (visible in URL after login)

PHASE        = 1                # 1 = first training | 2 = add new classes
PHASE1_PT    = "phase1_best.pt" # only used when PHASE = 2 (upload this to Colab)

# ── How to get correct workspace/project/version values ───────────────────────
# 1. Go to each Roboflow Universe URL below
# 2. Click "Fork Dataset" → copies to YOUR workspace
# 3. Click "Download Dataset" → YOLOv8 → "Show download code"
# 4. Copy the workspace/project/version from that snippet into PHASE1_DATASETS
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS       = 100
IMGSZ        = 640
BATCH        = 16
BASE_MODEL   = "yolov8n.pt"    # nano = best for mobile TFLite

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Install dependencies
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys

def install():
    print("📦 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "ultralytics", "roboflow", "pyyaml"])

install()

# ── imports after install ──────────────────────────────────────────────────────
import os, shutil, yaml
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Dataset definitions + class remapping
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (workspace, project_name, version, {original_class → target_class})
# Set target to None to DROP that class (e.g. generic "person" labels)
#
# ⚠️  If a download fails, check the exact workspace/project slug on Roboflow Universe.
#     Open the dataset URL → look at the URL path for the exact slug.

PHASE1_DATASETS = [
    # 🚬 Cigarettes / Smoking
    # Universe: https://universe.roboflow.com/kyunghee-university-ada5d/smoking-detection-3gefl
    (
        "kyunghee-university-ada5d",
        "smoking-detection-3gefl",
        4,
        {
            "cigarette":    "cigarette",
            "Cigarette":    "cigarette",
            "smoking":      "cigarette",
            "Smoking":      "cigarette",
            "smoke":        "cigarette",
            "cige":         "cigarette",   # alternate class name in this dataset
            "person":       None,          # drop
            "Person":       None,          # drop
            "adult":        None,          # drop
        },
    ),
    # 🪧 Mask / No-Mask
    # Actual classes: ['mask_weared_incorrect', 'with_mask', 'without_mask']
    (
        "gabriel-truong",
        "mask-detection-8cpsa",
        1,
        {
            "with_mask":            "mask",     # ← wearing mask correctly
            "without_mask":         "no_mask",  # ← no mask = violation
            "mask_weared_incorrect":"no_mask",  # ← incorrect mask = also violation
        },
    ),
    # 🔵 Gauges — classes in this dataset are ['center','face','tip'] (gauge parts, not gauge object)
    #            All dropped. Skip this dataset or replace with a better one.
    #            Commenting out to avoid wasting merge time.
    # (
    #     "dylan-kramp-gauge-training",
    #     "gauges-sfqxy",
    #     1,
    #     { "gauge": "gauge" },
    # ),
    # 🥽 Goggles / Safety glasses
    # Actual classes: ['Boots','Ear-Protection','Glasses','Gloves','Helmet','Mask','Vest']
    (
        "qsolalishankh",
        "safety-equipment-detector-l2ple",
        1,
        {
            "Glasses":          "goggles",   # ← fixed: was missing
            "Gloves":           "gloves",    # ← fixed: was missing
            "Mask":             "mask",      # ← fixed: was missing
            "Boots":            None,        # drop
            "Ear-Protection":   None,        # drop
            "Helmet":           None,        # drop
            "Vest":             None,        # drop
        },
    ),
    # 🧤 Gloves
    # Actual classes: ['construction-gloves']
    (
        "ppe-inb2b",
        "gloves-tpxn6",
        1,
        {
            "construction-gloves": "gloves",  # ← fixed: actual class name
        },
    ),
    # 🔥 Matches
    (
        "robotarm",
        "match-hxwnh",
        1,
        {
            "match":    "match",
            "Match":    "match",
            "matches":  "match",
        },
    ),
    # ⚙️  Compressor
    (
        "rohit-workspace",
        "compressor-with-corrosion",
        1,
        {
            "compressor":   "compressor",
            "Compressor":   "compressor",
        },
    ),
    # 🔥 Lighters
    (
        "freeze-mtroi",
        "detect-lighters",
        1,
        {
            "lighter":  "lighter",
            "Lighter":  "lighter",
            "lighters": "lighter",
        },
    ),
]

# ── Final unified class list ───────────────────────────────────────────────────
# Order matters — index = class ID baked into the trained model.
# Violations (🔴) first, compliant (🟢) after.
# Must match _violationClasses in yolo_detector.dart

PHASE1_CLASSES = [
    "cigarette",    # 0  🔴 violation
    "lighter",      # 1  🔴 violation
    "match",        # 2  🔴 violation
    "no_mask",      # 3  🔴 violation
    "no_goggles",   # 4  🔴 violation
    "no_gloves",    # 5  🔴 violation
    "mask",         # 6  🟢 compliant
    "goggles",      # 7  🟢 compliant
    "gloves",       # 8  🟢 compliant
    "compressor",   # 9  🟢 compliant
    "gauge",        # 10 🟢 compliant
]

# Phase 2 additions — collected from client photos on ship
PHASE2_EXTRA_CLASSES = [
    "solvent",          # 11 🔴 violation
    "spray_gun",        # 12 🔴 violation
    "two_pack_paint",   # 13 🔴 violation
    "rust_remover",     # 14 🔴 violation
    "msds",             # 15 🟢 compliant (Material Safety Data Sheet)
]

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Download datasets from Roboflow
# ══════════════════════════════════════════════════════════════════════════════

def download_datasets(rf: Roboflow, datasets: list, out_dir: Path) -> list:
    """Download each dataset, return list of (path, remap_dict)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for ws, proj, ver, remap in datasets:
        dest = out_dir / proj
        print(f"\n📥 Downloading  {ws}/{proj}  version {ver} ...")
        try:
            project = rf.workspace(ws).project(proj)
            dataset = project.version(ver).download("yolov8", location=str(dest))
            downloaded.append((Path(dataset.location), remap))
            print(f"   ✅ saved → {dest}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print(f"      Check workspace/project slug on Roboflow Universe.")

    return downloaded

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Merge all datasets into one with remapped class IDs
# ══════════════════════════════════════════════════════════════════════════════

def get_classes_from_yaml(dataset_path: Path) -> list:
    """Parse class names from a dataset's data.yaml."""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def merge_datasets(downloaded: list, global_classes: list, merged_dir: Path):
    """
    Merge all downloaded datasets into merged_dir.
    Remaps class indices according to each dataset's remap dict.
    Drops annotations whose class is not in the remap or maps to None.
    """
    merged_dir.mkdir(parents=True, exist_ok=True)
    global_idx = {cls: i for i, cls in enumerate(global_classes)}

    for split in ["train", "valid", "test"]:
        (merged_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images   = 0
    total_annots   = 0
    total_dropped  = 0

    for ds_path, remap in downloaded:
        try:
            ds_classes = get_classes_from_yaml(ds_path)
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {ds_path.name}: {e}")
            continue

        print(f"\n🔀 Merging  {ds_path.name}")
        print(f"   source classes : {ds_classes}")

        # Build old_index → new_index map (None = drop)
        idx_map = {}
        for old_idx, old_name in enumerate(ds_classes):
            target_name = remap.get(old_name)
            if target_name is None:
                idx_map[old_idx] = None
            elif target_name in global_idx:
                idx_map[old_idx] = global_idx[target_name]
            else:
                print(f"   ⚠️  '{old_name}' → '{target_name}' not in global classes — dropping")
                idx_map[old_idx] = None

        for split in ["train", "valid", "test"]:
            img_dir = ds_path / split / "images"
            lbl_dir = ds_path / split / "labels"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.iterdir()):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                new_lines = []
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        old_cls = int(parts[0])
                        new_cls = idx_map.get(old_cls)
                        if new_cls is None:
                            total_dropped += 1
                            continue
                        new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
                        total_annots += 1

                if not new_lines:
                    continue  # image has only dropped classes — skip

                # Use dataset name as prefix to avoid filename collisions
                stem = f"{ds_path.name}__{img_path.stem}"
                shutil.copy(img_path,
                            merged_dir / split / "images" / (stem + img_path.suffix))
                with open(merged_dir / split / "labels" / (stem + ".txt"), "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                total_images += 1

    print(f"\n✅ Merge complete")
    print(f"   Images     : {total_images}")
    print(f"   Annotations: {total_annots}")
    print(f"   Dropped    : {total_dropped}  (unmapped classes)")


def write_data_yaml(merged_dir: Path, classes: list) -> Path:
    content = {
        "path":  str(merged_dir.absolute()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(classes),
        "names": classes,
    }
    yaml_path = merged_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    print(f"📄 data.yaml → {yaml_path}")
    print(f"   {len(classes)} classes: {classes}")
    return yaml_path

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Train
# ══════════════════════════════════════════════════════════════════════════════

def train(data_yaml: Path, phase: int):
    last_pt   = Path(f"runs/detect/painting_safety_phase{phase}/weights/last.pt")
    best_pt   = Path(f"runs/detect/painting_safety_phase{phase}/weights/best.pt")

    if last_pt.exists():
        # ── RESUME from last checkpoint (interrupted session) ──────────────
        print(f"\n⏩ Resuming interrupted training from  {last_pt}")
        print(f"   Training will continue from the epoch it stopped at.")
        model = YOLO(str(last_pt))
        model.train(resume=True)   # reads all args from checkpoint — no need to repeat them

    elif phase == 2 and Path(PHASE1_PT).exists():
        # ── Phase 2 fine-tune from Phase 1 weights ─────────────────────────
        print(f"\n🔁 Phase 2 — fine-tuning from  {PHASE1_PT}")
        model = YOLO(PHASE1_PT)
        model.train(
            data        = str(data_yaml),
            epochs      = EPOCHS,
            imgsz       = IMGSZ,
            batch       = BATCH,
            name        = f"painting_safety_phase{phase}",
            patience    = 20,
            save        = True,
            save_period = 5,     # save checkpoint every 5 epochs as backup
            cache       = False,
            workers     = 2,
        )

    else:
        # ── Phase 1 fresh training ─────────────────────────────────────────
        if phase == 2:
            print(f"⚠️  {PHASE1_PT} not found — falling back to base model")
        print(f"\n🚀 Phase {phase} — starting from  {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        model.train(
            data        = str(data_yaml),
            epochs      = EPOCHS,
            imgsz       = IMGSZ,
            batch       = BATCH,
            name        = f"painting_safety_phase{phase}",
            patience    = 20,
            save        = True,
            save_period = 5,     # save checkpoint every 5 epochs as backup
            cache       = False,
            workers     = 2,
        )

    print(f"\n✅ Training complete")
    print(f"   best.pt  → {best_pt}")
    print(f"   last.pt  → {last_pt}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Export to TFLite (float32) for Flutter app
# ══════════════════════════════════════════════════════════════════════════════

def export_tflite(phase: int):
    best_pt = Path(f"runs/detect/painting_safety_phase{phase}/weights/best.pt")

    if not best_pt.exists():
        print(f"❌ {best_pt} not found — did training finish?")
        return

    print(f"\n📦 Exporting → TFLite float32 ...")
    model = YOLO(str(best_pt))
    model.export(format="tflite", imgsz=IMGSZ, int8=False)

    # Locate the generated .tflite file
    tflite_src = best_pt.parent / "best_float32.tflite"
    tflite_dst = Path(f"phase{phase}_painting_safety_float32.tflite")

    if tflite_src.exists():
        shutil.copy(tflite_src, tflite_dst)
        print(f"✅ TFLite  → {tflite_dst}")
    else:
        print(f"⚠️  Could not find {tflite_src} — check Ultralytics export output above")

    # Save .pt for next phase
    pt_dst = Path(f"phase{phase}_best.pt")
    shutil.copy(best_pt, pt_dst)
    print(f"✅ Weights → {pt_dst}  (use this as PHASE1_PT in Phase {phase + 1})")

    print(f"""
╔══════════════════════════════════════════════════════╗
║  NEXT STEPS                                          ║
╠══════════════════════════════════════════════════════╣
║  1. Download  {str(tflite_dst):<38}║
║  2. Copy to   assets/models/  in Flutter project     ║
║  3. Update model path in yolo_detector.dart          ║
║  4. Save      {str(pt_dst):<38}║
║     (needed for Phase {phase + 1} training)                     ║
╚══════════════════════════════════════════════════════╝
""")

    # Print updated _violationClasses for yolo_detector.dart
    print("── Update _violationClasses in yolo_detector.dart ──────────────")
    violation_classes = [c for c in (PHASE1_CLASSES + (PHASE2_EXTRA_CLASSES if phase == 2 else []))
                         if c.startswith("no_") or c in {"cigarette", "lighter", "match",
                                                          "solvent", "spray_gun",
                                                          "two_pack_paint", "rust_remover"}]
    print("static const _violationClasses = {")
    for c in violation_classes:
        print(f"  '{c}',")
    print("};")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  Painting Mantra Safety — YOLO Training  |  Phase {PHASE}")
    print("=" * 60)

    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Set ROBOFLOW_API_KEY at the top of this file first!")
        return

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    raw_dir    = Path("datasets_raw")
    merged_dir = Path("datasets_merged")

    classes = PHASE1_CLASSES if PHASE == 1 else PHASE1_CLASSES + PHASE2_EXTRA_CLASSES
    print(f"\n📋 {len(classes)} classes: {classes}")

    # ── Download ──
    downloaded = download_datasets(rf, PHASE1_DATASETS, raw_dir)
    if not downloaded:
        print("❌ No datasets downloaded — check API key and dataset slugs.")
        return

    # ── Merge ──
    merge_datasets(downloaded, classes, merged_dir)

    # ── data.yaml ──
    data_yaml = write_data_yaml(merged_dir, classes)

    # ── Train ──
    train(data_yaml, PHASE)

    # ── Export ──
    export_tflite(PHASE)


if __name__ == "__main__":
    main()
