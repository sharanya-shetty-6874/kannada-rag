"""
evaluation/recompute_metrics.py  — FINAL FIXED VERSION
═══════════════════════════════════════════════════════
Recomputes all metrics from saved JSON. No Groq. No DLL issues.

HOW TO RUN (from project root):
  python evaluation\recompute_metrics.py

It automatically finds results_combos.json and results_no_rag.json
in the evaluation\ folder and writes results_final.json + results_final.csv

WHY THE PREVIOUS VERSION FAILED:
  Problem 1: ROUGE = 0.000  → rouge_score library not installed on Windows
             FIX: Pure Python ROUGE below (zero dependencies)
  Problem 2: BERT = 0.000   → torch DLL (WinError 1114) crashed BERTScore
             FIX: Preserve existing correct BERT values from original JSON
  Problem 3: Recompute overwrote correct values with 0
             FIX: Only recompute ROUGE/METEOR/BLEU; keep stored BERT
"""

import os, sys, json, csv, unicodedata, re, math, argparse
from collections import Counter
from datetime import datetime
from typing import List, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_COMBOS = os.path.join(ROOT_DIR, "evaluation", "results_combos.json")
DEFAULT_NO_RAG = os.path.join(ROOT_DIR, "evaluation", "results_no_rag.json")
DEFAULT_OUT    = os.path.join(ROOT_DIR, "evaluation", "results_final.json")
DEFAULT_CSV    = os.path.join(ROOT_DIR, "evaluation", "results_final.csv")


# ══════════════════════════════════════════════════════════════
#  KANNADA TOKENIZER
# ══════════════════════════════════════════════════════════════

def normalize_kn(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[^\u0C80-\u0CFF\s]', ' ', text)
    return ' '.join(text.split()).strip()

def tokenize_kn(text: str) -> List[str]:
    return [w for w in normalize_kn(text).split() if w.strip()]


# ══════════════════════════════════════════════════════════════
#  PURE PYTHON ROUGE  (no rouge_score library needed)
# ══════════════════════════════════════════════════════════════

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _rouge_n(pred_tok, ref_tok, n):
    pg = _ngrams(pred_tok, n); rg = _ngrams(ref_tok, n)
    if not pg or not rg: return 0.0
    ov = sum((pg & rg).values())
    P  = ov / sum(pg.values()); R = ov / sum(rg.values())
    return round(2*P*R/(P+R), 4) if (P+R) else 0.0

def _lcs(x, y):
    m, n = len(x), len(y)
    prev = [0]*(n+1); curr = [0]*(n+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            curr[j] = prev[j-1]+1 if x[i-1]==y[j-1] else max(prev[j],curr[j-1])
        prev, curr = curr, [0]*(n+1)
    return prev[n]

def _rouge_l(pred_tok, ref_tok):
    if not pred_tok or not ref_tok: return 0.0
    l = _lcs(pred_tok, ref_tok)
    P = l/len(pred_tok); R = l/len(ref_tok)
    return round(2*P*R/(P+R), 4) if (P+R) else 0.0

def compute_rouge(pred: str, ref: str) -> Dict:
    p = tokenize_kn(pred); r = tokenize_kn(ref)
    return {
        "rouge1": _rouge_n(p, r, 1),
        "rouge2": _rouge_n(p, r, 2),
        "rougeL": _rouge_l(p, r),
    }


# ══════════════════════════════════════════════════════════════
#  METEOR  (paper formula: 10PR/(R+9P))
# ══════════════════════════════════════════════════════════════

def compute_meteor(pred: str, ref: str) -> float:
    p = tokenize_kn(pred); r = tokenize_kn(ref)
    if not p or not r: return 0.0
    m = len(set(p) & set(r))
    if not m: return 0.0
    P = m/len(p); R = m/len(r); d = R+9*P
    return round(10*P*R/d, 4) if d else 0.0


# ══════════════════════════════════════════════════════════════
#  BLEU  (character-level, pure Python)
# ══════════════════════════════════════════════════════════════

def compute_bleu(pred: str, ref: str) -> float:
    p = list(normalize_kn(pred)); r = list(normalize_kn(ref))
    if not p or not r: return 0.0
    scores = []
    for n in range(1, 5):
        pg = Counter(tuple(p[i:i+n]) for i in range(len(p)-n+1))
        rg = Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
        ov = sum((pg & rg).values()) if pg and rg else 0
        scores.append((ov+1)/(sum(pg.values())+1) if pg else 0.0)
    if any(s == 0 for s in scores): return 0.0
    bp  = 1.0 if len(p) >= len(r) else math.exp(1-len(r)/len(p))
    return round(bp * math.exp(sum(math.log(s) for s in scores)/4), 4)


# ══════════════════════════════════════════════════════════════
#  RECALL@K  (retrieval quality proxy)
# ══════════════════════════════════════════════════════════════

def compute_recall_at_k(retrieved: List[str], ref: str) -> float:
    if not retrieved or not ref: return 0.0
    rt = set(tokenize_kn(ref))
    dt = set(tokenize_kn(" ".join(retrieved)))
    return round(len(rt & dt)/len(rt), 4) if rt else 0.0


# ══════════════════════════════════════════════════════════════
#  RECOMPUTE ONE ROW
#  BERT: preserve existing non-zero values (already correct in combos)
# ══════════════════════════════════════════════════════════════

def recompute_row(row: Dict) -> Dict:
    if row.get("skipped", False):
        return row

    pred = row.get("answer", "")
    ref  = row.get("reference", "")
    ret  = row.get("retrieved_texts", [])

    rouge = compute_rouge(pred, ref)
    row.update({
        "rouge1":      rouge["rouge1"],
        "rouge2":      rouge["rouge2"],
        "rougeL":      rouge["rougeL"],
        "meteor":      compute_meteor(pred, ref),
        "bleu":        compute_bleu(pred, ref),
        "recall_at_k": compute_recall_at_k(ret, ref),
    })

    # Keep existing BERT if it was computed correctly (non-zero)
    # Only try to recompute if all zero (i.e. failed in original run)
    if row.get("bert_f1", 0.0) == 0.0:
        row = _try_bert(row, pred, ref)

    return row


def _try_bert(row: Dict, pred: str, ref: str) -> Dict:
    """Try to compute BERTScore. If DLL/torch fails, keep 0 (no crash)."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        path = _find_kn_bert()
        if not path:
            return row

        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mod = AutoModel.from_pretrained(path, local_files_only=True)
        mod.to("cpu").eval()

        def emb(text):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                h = mod(**enc).last_hidden_state.squeeze(0)[1:-1]
            n = h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return h / n

        ep = emb(normalize_kn(pred)); er = emb(normalize_kn(ref))
        if ep.shape[0] == 0 or er.shape[0] == 0: return row

        sim = torch.mm(ep, er.T)
        P   = sim.max(dim=1).values.mean().item()
        R   = sim.max(dim=0).values.mean().item()
        F1  = 2*P*R/(P+R) if (P+R) > 1e-9 else 0.0
        row["bert_p"]  = round(float(P),  4)
        row["bert_r"]  = round(float(R),  4)
        row["bert_f1"] = round(float(F1), 4)
    except Exception:
        pass  # DLL error or missing model — keep 0, no crash
    return row


def _find_kn_bert() -> str:
    for base in [os.path.join(ROOT_DIR,"models"),
                 os.path.join(ROOT_DIR,"backend","models")]:
        if not os.path.isdir(base): continue
        hub = os.path.join(base,"models--l3cube-pune--kannada-sentence-bert-nli","snapshots")
        if os.path.isdir(hub):
            for snap in sorted(os.listdir(hub)):
                p = os.path.join(hub, snap)
                if os.path.isfile(os.path.join(p,"config.json")): return p
        for c in [os.path.join(base,"l3cube-pune","kannada-sentence-bert-nli"),
                  os.path.join(base,"kannada-sentence-bert-nli")]:
            if os.path.isfile(os.path.join(c,"config.json")): return c
    return ""


# ══════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════

MK = ["rouge1","rouge2","rougeL","meteor","bleu","bert_p","bert_r","bert_f1","recall_at_k"]

def _avg(lst): return round(sum(lst)/len(lst),4) if lst else 0.0

def _agg(rows):
    v = [r for r in rows if not r.get("skipped",False)]
    if not v:
        return {**{k:0.0 for k in MK},
                "latency_ms":0.0,"throughput":0.0,"answered":0,"total":len(rows),"words_avg":0.0}
    res = {k: _avg([r.get(k,0.0) for r in v]) for k in MK}
    lats = [r["latency_ms"] for r in v if r.get("latency_ms",0)>0]
    al = _avg(lats)
    res.update({"latency_ms":al, "throughput":round(1.0/(al/1000),4) if al>0 else 0.0,
                "answered":len(v), "total":len(rows),
                "words_avg":_avg([len(r.get("answer","").split()) for r in v])})
    return res

def _fd(rows, d): return [r for r in rows if r.get("difficulty")==d]


# ══════════════════════════════════════════════════════════════
#  TABLE PRINTING
# ══════════════════════════════════════════════════════════════

W = 42

def _sec(t): print(f"\n{'═'*125}\n  {t}\n{'═'*125}")

def _hdr():
    print(f"  {'Model / Combo':<{W}} | Ans    | R1     | R2     | RL     | METEOR | BLEU   | BERT-P | BERT-R | BERT-F1 | R@k    | Lat(s)")
    print(f"  {'-'*W}-+-------+--------+--------+--------+--------+--------+--------+--------+---------+--------+-------")

def _row(lbl, a):
    print(f"  {lbl[:W]:<{W}} | "
          f"{int(a.get('answered',0))}/{int(a.get('total',0))}  | "
          f"{a.get('rouge1',0):.4f} | {a.get('rouge2',0):.4f} | "
          f"{a.get('rougeL',0):.4f} | {a.get('meteor',0):.4f} | "
          f"{a.get('bleu',0):.4f} | {a.get('bert_p',0):.4f} | "
          f"{a.get('bert_r',0):.4f} | {a.get('bert_f1',0):.4f}  | "
          f"{a.get('recall_at_k',0):.4f} | {a.get('latency_ms',0)/1000:.2f}s")


def print_all_tables(combos: Dict, no_rag: Dict):
    diffs = ["easy","medium","hard"]
    dtab  = {"easy":"TABLE 4","medium":"TABLE 5","hard":"TABLE 6"}

    # TABLE 2
    _sec("TABLE 2 — Speed and Throughput")
    print(f"  {'Model / Combo':<{W}} | Avg Words | Latency    | Throughput (q/s) | Answered")
    print(f"  {'-'*W}-+-----------+------------+------------------+---------")
    for lbl,rows in {**combos,**no_rag}.items():
        a = _agg(rows)
        print(f"  {lbl[:W]:<{W}} | {a['words_avg']:9.1f} | "
              f"{a['latency_ms']/1000:8.2f}s   | {a['throughput']:16.4f} | "
              f"{int(a['answered'])}/{int(a['total'])}")

    # TABLES 4/5/6
    for diff in diffs:
        _sec(f"{dtab[diff]} — {diff.upper()} Level Questions")
        _hdr()
        items = []
        for lbl,rows in combos.items():
            f = _fd(rows,diff)
            if f: items.append((lbl,_agg(f)))
        for lbl,rows in no_rag.items():
            f = _fd(rows,diff)
            if f: items.append((f"[NoRAG] {lbl}",_agg(f)))
        for lbl,a in sorted(items,key=lambda x:-x[1].get("bert_f1",0)):
            _row(lbl,a)

    # TABLE 7
    _sec("TABLE 7 — Full Evaluation Matrix (All 25 Questions)")
    _hdr()
    all_m = [(lbl,_agg(rows)) for lbl,rows in combos.items()]
    all_m += [(f"[NoRAG] {lbl}",_agg(rows)) for lbl,rows in no_rag.items()]
    all_m.sort(key=lambda x:-x[1].get("bert_f1",0))
    for lbl,a in all_m: _row(lbl,a)

    # TABLE 9
    if combos and no_rag:
        _sec("TABLE 9 — RAG vs No-RAG BERT-F1 Comparison")
        best_r = max(combos.items(),  key=lambda kv:_agg(kv[1]).get("bert_f1",0))
        best_n = max(no_rag.items(),  key=lambda kv:_agg(kv[1]).get("bert_f1",0))
        print(f"\n  {'Difficulty':<10} | {'RAG BF1':<12} | {'No-RAG BF1':<13} | Improvement")
        print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*13}-+-----------")
        for diff in diffs:
            rf=_fd(best_r[1],diff); nf=_fd(best_n[1],diff)
            if rf and nf:
                rb=_agg(rf)["bert_f1"]; nb=_agg(nf)["bert_f1"]
                print(f"  {diff.upper():<10} | {rb:<12.4f} | {nb:<13.4f} | +{rb-nb:.4f}")
        ra=_agg(best_r[1])["bert_f1"]; na=_agg(best_n[1])["bert_f1"]
        print(f"  {'OVERALL':<10} | {ra:<12.4f} | {na:<13.4f} | +{ra-na:.4f}")
        print(f"\n  Best RAG  : {best_r[0]}\n  No-RAG    : {best_n[0]}")

    # LEADERBOARD
    _sec("LEADERBOARD — Ranked by BERT-F1 (primary metric)")
    print(f"\n  # | {'Model / Combo':<{W}} | BF1    | R1     | RL     | METEOR | BLEU   | Lat(s)")
    print(f"  --+-{'-'*W}-+--------+--------+--------+--------+--------+------")
    for i,(lbl,a) in enumerate(all_m,1):
        tag = "  ◀ BEST" if i==1 else ""
        print(f"  {i:<2} | {lbl[:W]:<{W}} | "
              f"{a['bert_f1']:.4f} | {a['rouge1']:.4f} | "
              f"{a['rougeL']:.4f} | {a['meteor']:.4f} | "
              f"{a['bleu']:.4f} | {a['latency_ms']/1000:.2f}s{tag}")

    # RECOMMENDATION
    if all_m:
        bl,ba = all_m[0]
        _sec("BEST MODEL RECOMMENDATION")
        print(f"\n  ✅  Best : {bl}\n")
        for k,v in [("BERT-F1",ba['bert_f1']),("ROUGE-1",ba['rouge1']),
                    ("ROUGE-2",ba['rouge2']),("ROUGE-L",ba['rougeL']),
                    ("METEOR",ba['meteor']),("BLEU",ba['bleu']),
                    ("Recall@k",ba['recall_at_k']),
                    ("Latency",f"{ba['latency_ms']/1000:.2f}s"),
                    ("Throughput",f"{ba['throughput']:.4f} q/s")]:
            print(f"  {k:<16}: {v}")
        if combos and no_rag:
            ra = ba["bert_f1"]
            na = max((_agg(rows)["bert_f1"] for rows in no_rag.values()), default=0)
            if na > 0:
                print(f"\n  RAG vs No-RAG : +{ra-na:.4f} BERT-F1  ({round((ra-na)/na*100,1)}% improvement)")

    print(f"\n{'═'*125}\n  ✅  Done!\n{'═'*125}\n")


# ══════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════

def export_csv(combos, no_rag, csv_path):
    fields = ["model","type","difficulty","answered","total",
              "rouge1","rouge2","rougeL","meteor","bleu",
              "bert_p","bert_r","bert_f1","recall_at_k",
              "latency_ms","throughput","words_avg"]

    rows_out = []
    def add(lbl, rows, mtype):
        for diff in ["easy","medium","hard","overall"]:
            fr = rows if diff=="overall" else _fd(rows,diff)
            if not fr: continue
            a = _agg(fr)
            rows_out.append({"model":lbl,"type":mtype,"difficulty":diff,
                             "answered":int(a["answered"]),"total":int(a["total"]),
                             **{k:a.get(k,0.0) for k in fields[5:]}})

    for lbl,rows in combos.items(): add(lbl,rows,"RAG")
    for lbl,rows in no_rag.items(): add(lbl,rows,"NoRAG")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path,"w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)
    print(f"  📊 CSV → {csv_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--combos", default=DEFAULT_COMBOS)
    p.add_argument("--no_rag", default=DEFAULT_NO_RAG)
    p.add_argument("--out",    default=DEFAULT_OUT)
    p.add_argument("--csv",    default=DEFAULT_CSV)
    args = p.parse_args()

    print("\n" + "═"*70)
    print("  Kannada RAG — Metric Recompute (FINAL FIXED)")
    print("  ROUGE : pure Python (no rouge_score library)")
    print("  BERT  : preserved from original JSON (no DLL needed)")
    print("═"*70)

    # Load
    combos_raw = {}
    if os.path.exists(args.combos):
        with open(args.combos,"r",encoding="utf-8") as f:
            combos_raw = json.load(f).get("combos",{})
        print(f"\n✅ Combos : {args.combos}  ({len(combos_raw)} models)")
    else:
        print(f"\n⚠️  Not found: {args.combos}")

    no_rag_raw = {}
    if os.path.exists(args.no_rag):
        with open(args.no_rag,"r",encoding="utf-8") as f:
            d = json.load(f)
            no_rag_raw = d.get("no_rag", d.get("combos",{}))
        print(f"✅ No-RAG : {args.no_rag}  ({len(no_rag_raw)} models)")
    else:
        print(f"⚠️  Not found: {args.no_rag}")

    print(f"\n📋 Total rows: {sum(len(v) for v in combos_raw.values()) + sum(len(v) for v in no_rag_raw.values())}")

    # Status check
    print("\n🔍 BERT status in stored JSON:")
    for lbl,rows in combos_raw.items():
        v=[r for r in rows if not r.get("skipped")]
        nz=[r for r in v if r.get("bert_f1",0)>0]
        print(f"   {lbl[:55]}: {len(nz)}/{len(v)} non-zero  → {'PRESERVING ✅' if nz else 'will attempt recompute'}")
    for lbl,rows in no_rag_raw.items():
        v=[r for r in rows if not r.get("skipped")]
        nz=[r for r in v if r.get("bert_f1",0)>0]
        print(f"   {lbl[:55]}: {len(nz)}/{len(v)} non-zero  → {'PRESERVING ✅' if nz else 'will attempt recompute'}")

    # Recompute
    print("\n🔄 Recomputing ROUGE / METEOR / BLEU / Recall@k …\n")

    new_combos = {}
    for lbl,rows in combos_raw.items():
        new_rows = [recompute_row(dict(r)) for r in rows]
        new_combos[lbl] = new_rows
        v=[r for r in new_rows if not r.get("skipped")]
        if v:
            print(f"  {lbl[:55]}")
            print(f"    R1={_avg([r['rouge1'] for r in v]):.4f}  "
                  f"RL={_avg([r['rougeL'] for r in v]):.4f}  "
                  f"MT={_avg([r['meteor'] for r in v]):.4f}  "
                  f"BLEU={_avg([r['bleu'] for r in v]):.4f}  "
                  f"BF1={_avg([r['bert_f1'] for r in v]):.4f}  "
                  f"({len(v)}/25)")

    new_no_rag = {}
    for lbl,rows in no_rag_raw.items():
        clean = lbl if lbl.startswith("NoRAG") else f"NoRAG|{lbl}"
        new_rows = [recompute_row(dict(r)) for r in rows]
        new_no_rag[clean] = new_rows
        v=[r for r in new_rows if not r.get("skipped")]
        if v:
            print(f"  {clean[:55]}")
            print(f"    R1={_avg([r['rouge1'] for r in v]):.4f}  "
                  f"RL={_avg([r['rougeL'] for r in v]):.4f}  "
                  f"MT={_avg([r['meteor'] for r in v]):.4f}  "
                  f"BLEU={_avg([r['bleu'] for r in v]):.4f}  "
                  f"BF1={_avg([r['bert_f1'] for r in v]):.4f}  "
                  f"({len(v)}/25)")

    # Save
    output = {"meta":{"recomputed":True,"timestamp":datetime.now().isoformat(),
                      "rouge_method":"pure_python",
                      "bert_method":"preserved_from_original_json"},
              "combos":new_combos, "no_rag":new_no_rag}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
    print(f"\n  💾 JSON → {args.out}")
    export_csv(new_combos, new_no_rag, args.csv)

    print_all_tables(new_combos, new_no_rag)