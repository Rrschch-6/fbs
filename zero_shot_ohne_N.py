import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import util   # <-- your util.py

# --------------------
# Paths & Params
# --------------------
root_file = "/home/sascha/fbs"
os.makedirs(f"{root_file}/results", exist_ok=True)
os.makedirs(f"{root_file}/responses", exist_ok=True)

xls_path = f"{root_file}/data/P04_Pen_Design_Protocol.xlsx"

max_new_tokens = 16
batch_size = 64

# Models
models = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]

# --------------------
# Load Data
# --------------------
df = pd.read_excel(xls_path)

# 🔹 Drop rows with N in ground truth, since we no longer want that class
df = df[df["CODE"] != "N"]

texts = df["UTTERANCE"].astype(str).tolist()
labels = df["CODE"].astype(str).tolist()

# Candidate labels (N removed)
CANDIDATE_LABELS = ["F", "Be", "Bs", "S", "D"]
LABEL_PATTERN = re.compile(r'(?<![A-Za-z])(F|Be|Bs|S|D)(?![A-Za-z])')

SYN_TO_LABEL = {
    "function": "F", "purpose": "F", "goal": "F",
    "expected behaviour": "Be", "expected behavior": "Be",
    "behaviour from structure": "Bs", "behavior from structure": "Bs",
    "structure": "S",
    "discussion": "D", "design rationale": "D"
}

# --------------------
# Prompt
# --------------------
GUIDE = """F–Be–Bs–S Labelling Guide (pilot)

Assign one label per segment.
Allowed labels: F, Be, Bs, S, D.

F — Function (purpose/goal)
  Cues: 'so that', 'to + verb', 'for + -ing', 'goal is', 'needs to'
  Examples: 'It needs to print legibly at 6pt.'; 'Designed to draw thin technical lines.'

Be — Expected Behaviour (performance independent of specific parts/materials)
  Cues: smoothly, reliably, quickly, consistently; no explicit part/material reference
  Examples: 'Writes smoothly and doesn’t skip.'; 'Heats up quickly.'

Bs — Behaviour from Structure (behaviour explicitly caused by a part/material/geometry/layout)
  Cues: 'because of', 'due to', 'with [part]', 'causes', explicit part→effect
  Examples: 'Steel barrel adds weight so lines are steadier.'; 'Porous tip wicks too fast so ink spreads.'

S — Structure (materials, parts, configuration), no behaviour claim
  Cues: part names, materials, counts, dimensions, assembly
  Examples: 'Steel barrel with replaceable cartridge.'; 'Two screws secure the feed roller.'

Disambiguation:
- Be vs Bs: Behaviour without part/material link → Be. With explicit part/material cause → Bs.
- S vs Bs: Pure description → S. Description that explains behaviour → Bs.
- F vs Be/Bs: 'Why/purpose' → F; 'how it behaves' → Be/Bs.
- If mixed, choose the dominant idea. If inseparable with clear cause→effect via parts, prefer Bs.

Note: The label set also includes 'D' (present in the data).

Instructions: Choose exactly ONE label from [F, Be, Bs, S, D].
Return your answer as JSON: {"label": "<F|Be|Bs|S|D>"} and nothing else.
"""

def build_prompt(text: str):
    return f"""{GUIDE}

Segment:
{text}

JSON answer:"""

# --------------------
# Label Parsing
# --------------------
def parse_label(generated: str) -> str:
    if not generated:
        return "D"   # fallback (instead of N)

    text = generated.strip()

    # JSON first
    try:
        json_blocks = re.findall(r'\{.*?\}', text, flags=re.DOTALL)
        for jb in reversed(json_blocks):
            try:
                obj = json.loads(jb)
                lbl = obj.get("label", "").strip()
                if lbl in CANDIDATE_LABELS:
                    return lbl
            except Exception:
                continue
    except Exception:
        pass

    # "Label:" lines
    m = re.search(r'(?:Final\s*Label|Label)\s*[:\-]\s*(F|Be|Bs|S|D)\b', text, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Regex (last match)
    matches = list(LABEL_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1)

    # Synonyms
    lower = text.lower()
    for syn, lab in SYN_TO_LABEL.items():
        if syn in lower:
            return lab

    return "D"   # fallback

# --------------------
# Main Eval Loop
# --------------------
results = []

for model_id in models:
    print(f"\n=== Evaluating {model_id} ===")
    model, tokenizer = util.load_model(model_id, device="auto")

    prompts = [build_prompt(t) for t in texts]
    outputs = util.generate_output(
        model, tokenizer, prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        model_name=model_id.split("/")[1]
    )

    preds = [parse_label(out) for out in outputs]

    # Save per-model responses
    resp_df = pd.DataFrame({
        "UTTERANCE": texts,
        "TRUE_LABEL": labels,
        "RESPONSE": outputs,
        "PRED_LABEL": preds
    })
    resp_path = f"{root_file}/responses/responses_{model_id.replace('/', '_')}.csv"
    resp_df.to_csv(resp_path, index=False)

    # Metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=CANDIDATE_LABELS, average="weighted", zero_division=0
    )

    results.append({
        "model": model_id,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    cm = confusion_matrix(labels, preds, labels=CANDIDATE_LABELS)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_id}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CANDIDATE_LABELS)))
    ax.set_yticks(range(len(CANDIDATE_LABELS)))
    ax.set_xticklabels(CANDIDATE_LABELS)
    ax.set_yticklabels(CANDIDATE_LABELS)

    # Annotate cells
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Add accuracy and F1 below the x-axis
    metrics_text = f"Accuracy: {acc:.3f}   |   F1: {f1:.3f}"
    fig.text(0.5, -0.05, metrics_text, ha="center", fontsize=12, style="italic")

    fig.tight_layout()
    fig.savefig(f"{root_file}/results/confusion_{model_id.replace('/', '_')}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Save metrics
results_df = pd.DataFrame(results)
results_df.to_csv(f"{root_file}/results/metrics.csv", index=False)

print("\n✓ Done. Results + responses saved in:", f"{root_file}/results and {root_file}/responses")
