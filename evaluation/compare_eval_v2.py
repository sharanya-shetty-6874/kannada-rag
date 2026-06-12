"""
evaluation/compare_eval_v2.py
═══════════════════════════════════════════════════════════════════
COMPLETE Kannada RAG Evaluation — Research Paper Quality
═══════════════════════════════════════════════════════════════════

Tables produced (matching published Kannada RAG papers):
  Table 2  — Speed & Throughput by model
  Table 4  — Easy-level evaluation results
  Table 5  — Medium-level evaluation results
  Table 6  — Hard-level evaluation results
  Table 7  — Full evaluation matrix (all metrics × all combos)
  Table 9  — RAG vs No-RAG BERT-F1 comparison

Metrics:
  ROUGE-1, ROUGE-2, ROUGE-L  (Kannada-aware tokenizer — FIXED)
  METEOR  (simplified formula matching the paper: (10×P×R)/(R+9×P))
  BLEU    (character-level)
  BERTScore P / R / F1  (local KN-BERT, no internet)
  Recall@k  (retrieval quality proxy)
  Latency (ms) + Throughput (q/s)

Outputs:
  results_<ts>.json  — raw per-question data
  results_<ts>.csv   — summary for Excel/report
  Printed tables + best-model recommendation

USAGE:
  python evaluation/compare_eval_v2.py --pdf Kan-Science --gt evaluation/kannada_ground_truth_25.json --mode all
  python evaluation/compare_eval_v2.py --pdf Kan-Science --gt evaluation/kannada_ground_truth_25.json --mode combos_only
  python evaluation/compare_eval_v2.py --pdf Kan-Science --gt evaluation/kannada_ground_truth_25.json --mode no_rag_only
  python evaluation/compare_eval_v2.py --pdf Kan-Science --gt evaluation/kannada_ground_truth_25.json --mode llm_compare
  python evaluation/compare_eval_v2.py --all_combos --pdf Kan-Science --gt evaluation/kannada_ground_truth_25.json
  python evaluation/compare_eval_v2.py --print_only evaluation/results_v2_20260422_1400.json
"""

import os, sys, json, csv, time, argparse, unicodedata, re
from datetime import datetime
from typing import List, Dict, Optional

os.environ["HF_HUB_OFFLINE"]      = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.compare.compare_orchestrator import (
    ALL_COMBOS, LLM_COMPARISON_COMBOS, NO_RAG_MODELS,
    run_combo, run_no_rag, ComboConfig,
)


# ══════════════════════════════════════════════════════════════
#  KANNADA TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

def normalize_kannada(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[^\u0C80-\u0CFF\s\u0964\u0965,.!?]', ' ', text)
    return ' '.join(text.split()).strip()


def tokenize_kannada(text: str) -> List[str]:
    return [w for w in normalize_kannada(text).split() if w.strip()]


# ══════════════════════════════════════════════════════════════
#  ROUGE  (FIXED for Kannada Unicode)
# ══════════════════════════════════════════════════════════════

def compute_rouge(pred: str, ref: str) -> Dict[str, float]:
    """
    FIXED: Pre-tokenize with Kannada-aware tokenizer before rouge_score.
    Default tokenizer treats the entire Kannada string as one token → 0.000.
    Correct order: scorer.score(target=reference, prediction=generated)
    """
    try:
        from rouge_score import rouge_scorer
        p = " ".join(tokenize_kannada(pred))
        r = " ".join(tokenize_kannada(ref))
        if not p or not r:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
        res = sc.score(target=r, prediction=p)   # target=ref FIRST — critical
        return {k: round(res[k].fmeasure, 4) for k in ("rouge1","rouge2","rougeL")}
    except Exception as e:
        print(f"  ⚠️  ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ══════════════════════════════════════════════════════════════
#  METEOR  (paper formula, Kannada token overlap)
# ══════════════════════════════════════════════════════════════

def compute_meteor(pred: str, ref: str) -> float:
    """
    METEOR = (10 × P × R) / (R + 9 × P)
    Same simplified formula used in the other team's paper.
    NLTK wordnet does not support Kannada, so we use token overlap.
    """
    pt = tokenize_kannada(pred)
    rt = tokenize_kannada(ref)
    if not pt or not rt:
        return 0.0
    m = len(set(pt) & set(rt))
    if m == 0:
        return 0.0
    P = m / len(pt)
    R = m / len(rt)
    d = R + 9 * P
    return round((10 * P * R) / d, 4) if d else 0.0


# ══════════════════════════════════════════════════════════════
#  BLEU  (character-level)
# ══════════════════════════════════════════════════════════════

def compute_bleu(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        p = normalize_kannada(pred)
        r = normalize_kannada(ref)
        return round(sentence_bleu([list(r)], list(p),
                     smoothing_function=SmoothingFunction().method1), 4)
    except Exception as e:
        print(f"  ⚠️  BLEU: {e}")
        return 0.0


# ══════════════════════════════════════════════════════════════
#  BERTScore  (local KN-BERT model, no internet required)
# ══════════════════════════════════════════════════════════════

_BS_ENABLED   = True
_BS_LOADED    = False
_bs_tok = _bs_mod = _bs_dev = None


def _find_local_bert() -> str:
    base = os.path.join(ROOT_DIR, "models")
    hub  = os.path.join(base, "models--l3cube-pune--kannada-sentence-bert-nli", "snapshots")
    if os.path.isdir(hub):
        for snap in sorted(os.listdir(hub)):
            p = os.path.join(hub, snap)
            if os.path.isfile(os.path.join(p, "config.json")):
                return p
    for c in [os.path.join(base,"l3cube-pune","kannada-sentence-bert-nli"),
              os.path.join(base,"kannada-sentence-bert-nli")]:
        if os.path.isfile(os.path.join(c,"config.json")):
            return c
    return ""


_LOCAL_BERT = _find_local_bert()


def _load_bert() -> bool:
    global _BS_ENABLED, _BS_LOADED, _bs_tok, _bs_mod, _bs_dev
    if _BS_LOADED:
        return _BS_ENABLED
    _BS_LOADED = True
    if not _LOCAL_BERT:
        _BS_ENABLED = False
        print("  ⚠️  KN-BERT not found — BERTScore disabled")
        return False
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"  📥 BERTScore model: {_LOCAL_BERT}")
        _bs_tok = AutoTokenizer.from_pretrained(_LOCAL_BERT, local_files_only=True)
        _bs_mod = AutoModel.from_pretrained(_LOCAL_BERT, local_files_only=True)
        _bs_dev = "cuda" if torch.cuda.is_available() else "cpu"
        _bs_mod.to(_bs_dev).eval()
        print(f"  ✅ BERTScore ready on {_bs_dev}")
        return True
    except Exception as e:
        _BS_ENABLED = False
        print(f"  ⚠️  BERTScore load failed: {e}")
        return False


def _tok_emb(text: str):
    import torch
    enc = _bs_tok(text, return_tensors="pt", truncation=True,
                  max_length=512, padding=False)
    enc = {k: v.to(_bs_dev) for k, v in enc.items()}
    with torch.no_grad():
        h = _bs_mod(**enc).last_hidden_state.squeeze(0)[1:-1]
    n = h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return h / n


def compute_bertscore(pred: str, ref: str) -> Dict[str, float]:
    global _BS_ENABLED
    if not _load_bert():
        return {"bert_p": 0.0, "bert_r": 0.0, "bert_f1": 0.0}
    try:
        import torch
        ep = _tok_emb(normalize_kannada(pred))
        er = _tok_emb(normalize_kannada(ref))
        if ep.shape[0] == 0 or er.shape[0] == 0:
            return {"bert_p": 0.0, "bert_r": 0.0, "bert_f1": 0.0}
        sim = torch.mm(ep, er.T)
        P   = sim.max(dim=1).values.mean().item()
        R   = sim.max(dim=0).values.mean().item()
        F1  = 2*P*R/(P+R) if (P+R) > 1e-9 else 0.0
        return {"bert_p": round(P,4), "bert_r": round(R,4), "bert_f1": round(F1,4)}
    except Exception as e:
        _BS_ENABLED = False
        print(f"  ⚠️  BERTScore compute: {e}")
        return {"bert_p": 0.0, "bert_r": 0.0, "bert_f1": 0.0}
    finally:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ══════════════════════════════════════════════════════════════
#  RECALL@K  (retrieval quality proxy)
# ══════════════════════════════════════════════════════════════

def compute_recall_at_k(retrieved_texts: List[str], ref: str) -> float:
    """
    Fraction of reference tokens found anywhere in the retrieved context.
    Proxy for Recall@k without needing explicit relevance labels.
    """
    if not retrieved_texts or not ref:
        return 0.0
    ref_tok  = set(tokenize_kannada(ref))
    ret_tok  = set(tokenize_kannada(" ".join(retrieved_texts)))
    if not ref_tok:
        return 0.0
    return round(len(ref_tok & ret_tok) / len(ref_tok), 4)


# ══════════════════════════════════════════════════════════════
#  ALL METRICS
# ══════════════════════════════════════════════════════════════

MK = ["rouge1","rouge2","rougeL","meteor","bleu","bert_p","bert_r","bert_f1","recall_at_k"]


def all_metrics(pred: str, ref: str, retrieved: List[str] = None) -> Dict:
    r = compute_rouge(pred, ref)
    b = compute_bertscore(pred, ref)
    return {
        "rouge1": r["rouge1"], "rouge2": r["rouge2"], "rougeL": r["rougeL"],
        "meteor": compute_meteor(pred, ref),
        "bleu":   compute_bleu(pred, ref),
        "bert_p": b["bert_p"], "bert_r": b["bert_r"], "bert_f1": b["bert_f1"],
        "recall_at_k": compute_recall_at_k(retrieved or [], ref),
    }


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

_ERR = ("⚠️","⏳","🚫")

def _is_err(a: str) -> bool:
    return not a or any(a.strip().startswith(p) for p in _ERR)

def _blank(qid,diff,q,ref,ans,lat):
    return {"id":qid,"difficulty":diff,"question":q,"reference":ref,
            "answer":ans,"skipped":True,"latency_ms":lat,
            "confidence":0.0,"hallucination_risk":"low","words_in_answer":0,
            **{k:0.0 for k in MK}}

def _avg(lst): return round(sum(lst)/len(lst),4) if lst else 0.0

def _agg(rows: List[Dict]) -> Dict:
    v = [r for r in rows if not r.get("skipped",False)]
    if not v:
        return {**{k:0.0 for k in MK},
                "latency_ms":0.0,"throughput":0.0,"answered":0,"total":len(rows),"words_avg":0.0}
    res = {k: _avg([r.get(k,0.0) for r in v]) for k in MK}
    lats = [r["latency_ms"] for r in v if r.get("latency_ms",0)>0]
    al   = _avg(lats)
    res["latency_ms"]  = al
    res["throughput"]  = round(1.0/(al/1000),4) if al>0 else 0.0
    res["answered"]    = len(v)
    res["total"]       = len(rows)
    res["words_avg"]   = _avg([r.get("words_in_answer",0) for r in v])
    return res

def _fd(rows, diff): return [r for r in rows if r.get("difficulty")==diff]

# Sleep config (Groq free tier)
_SL_S = 3.0    # small LLM (8b/9b/7b)
_SL_L = 1.5    # large LLM (70b)
_SL_NR= 2.0    # no-RAG
_SL_GAP= 8.0   # between combos


# ══════════════════════════════════════════════════════════════
#  EVALUATION LOOPS
# ══════════════════════════════════════════════════════════════

def eval_combos(gt: List[Dict], pdf: str, combos: List[ComboConfig]) -> Dict[str,List]:
    total = len(combos)*len(gt); call_n = 0; out = {}
    print(f"\n{'━'*70}")
    print(f"  🔬 COMBO EVALUATION — {len(combos)} combos × {len(gt)} questions = {total} calls")
    print(f"{'━'*70}")

    for ci, combo in enumerate(combos):
        sl   = _SL_S if combo.is_small_llm else _SL_L
        rows = []
        print(f"\n  [{ci+1}/{len(combos)}] {combo.label}")
        print(f"  {'─'*67}")

        for qi, item in enumerate(gt):
            call_n += 1
            qid = item["id"]; q = item["question"]; ref = item["ground_truth_answer"]
            diff = item.get("difficulty","medium")
            print(f"  [{call_n:>3}/{total}] {diff[:3].upper()} {q[:44]}…", end="", flush=True)

            res  = run_combo(q, pdf, combo)
            pred = res.get("answer","")

            if _is_err(pred):
                print(" SKIP")
                rows.append(_blank(qid,diff,q,ref,pred,res.get("latency_ms",0)))
                time.sleep(sl); continue

            m = all_metrics(pred, ref, res.get("retrieved_texts",[]))
            print(f" R1={m['rouge1']:.3f} BF={m['bert_f1']:.3f} MT={m['meteor']:.3f} "
                  f"R@k={m['recall_at_k']:.3f} {res.get('latency_ms',0):.0f}ms")
            rows.append({"id":qid,"difficulty":diff,"question":q,"reference":ref,
                         "answer":pred,"skipped":False,
                         "latency_ms":res.get("latency_ms",0),
                         "confidence":res.get("confidence",0),
                         "hallucination_risk":res.get("hallucination_risk","low"),
                         "words_in_answer":len(pred.split()), **m})
            if qi < len(gt)-1: time.sleep(sl)

        out[combo.label] = rows
        a = _agg(rows)
        print(f"\n  📊 {a['answered']}/{a['total']} | "
              f"R1={a['rouge1']:.3f} RL={a['rougeL']:.3f} "
              f"BF1={a['bert_f1']:.3f} MT={a['meteor']:.3f} Lat={a['latency_ms']:.0f}ms")
        if ci < len(combos)-1:
            print(f"  💤 {_SL_GAP:.0f}s pause…"); time.sleep(_SL_GAP)
    return out


def eval_no_rag(gt: List[Dict], models: List[str]) -> Dict[str,List]:
    out = {}
    print(f"\n{'━'*70}")
    print(f"  🤖 NO-RAG BASELINE — {models}")
    print(f"{'━'*70}")

    for model in models:
        sl   = _SL_S if any(t in model.lower() for t in ("8b","9b","7b")) else _SL_NR
        rows = []
        print(f"\n  Model: {model}")
        print(f"  {'─'*67}")

        for qi, item in enumerate(gt):
            qid = item["id"]; q = item["question"]; ref = item["ground_truth_answer"]
            diff = item.get("difficulty","medium")
            print(f"  [{qi+1:>2}/{len(gt)}] {diff[:3].upper()} {q[:44]}…", end="", flush=True)

            res  = run_no_rag(q, llm_model=model)
            pred = res.get("answer","")
            if _is_err(pred) or res.get("error"):
                print(" SKIP"); rows.append(_blank(qid,diff,q,ref,pred,res.get("latency_ms",0)))
                time.sleep(sl); continue

            m = all_metrics(pred, ref, [])
            print(f" R1={m['rouge1']:.3f} BF={m['bert_f1']:.3f} MT={m['meteor']:.3f} {res.get('latency_ms',0):.0f}ms")
            rows.append({"id":qid,"difficulty":diff,"question":q,"reference":ref,
                         "answer":pred,"skipped":False,
                         "latency_ms":res.get("latency_ms",0),
                         "confidence":0.0,"hallucination_risk":"unknown",
                         "words_in_answer":len(pred.split()), **m})
            if qi < len(gt)-1: time.sleep(sl)

        out[f"NoRAG|{model}"] = rows
        a = _agg(rows)
        print(f"\n  📊 {a['answered']}/{a['total']} | R1={a['rouge1']:.3f} BF1={a['bert_f1']:.3f}")
    return out


# ══════════════════════════════════════════════════════════════
#  TABLE PRINTING
# ══════════════════════════════════════════════════════════════

W = 36

def _sec(t): print(f"\n{'═'*122}\n  {t}\n{'═'*122}")

def _hdr():
    print(f"  {'Model/Combo':<{W}} | Ans  | R1     | R2     | RL     | METEOR | BLEU   | BERT-P | BERT-R | BERT-F1 | R@k    | Lat(ms)  | Q/s")
    print(f"  {'-'*W}-+-----+--------+--------+--------+--------+--------+--------+--------+---------+--------+----------+------")

def _row(lbl, a):
    ans = f"{int(a.get('answered',0))}/{int(a.get('total',0))}"
    print(f"  {lbl[:W]:<{W}} | {ans:<5}| "
          f"{a.get('rouge1',0):.4f} | {a.get('rouge2',0):.4f} | "
          f"{a.get('rougeL',0):.4f} | {a.get('meteor',0):.4f} | "
          f"{a.get('bleu',0):.4f} | {a.get('bert_p',0):.4f} | "
          f"{a.get('bert_r',0):.4f} | {a.get('bert_f1',0):.4f}  | "
          f"{a.get('recall_at_k',0):.4f} | {a.get('latency_ms',0):8.1f} | {a.get('throughput',0):.4f}")


def generate_all_tables(output: Dict):
    CR  = output.get("combos",{})
    NR  = output.get("no_rag",{})
    diffs = ["easy","medium","hard"]
    dtab  = {"easy":"TABLE 4","medium":"TABLE 5","hard":"TABLE 6"}

    # TABLE 2 — Speed
    _sec("TABLE 2 — Speed and Throughput")
    print(f"  {'Model/Combo':<{W}} | Avg Words | Avg Latency(ms) | Q/s    | Answered")
    print(f"  {'-'*W}-+-----------+-----------------+--------+---------")
    for lbl,rows in {**CR,**NR}.items():
        a = _agg(rows)
        print(f"  {lbl[:W]:<{W}} | {a['words_avg']:9.2f} | {a['latency_ms']:15.1f} | {a['throughput']:6.4f} | {int(a['answered'])}/{int(a['total'])}")

    # TABLE 4/5/6 — Per-Difficulty
    for diff in diffs:
        _sec(f"{dtab[diff]} — {diff.upper()} Questions")
        _hdr()
        rows_list = [(lbl, _agg(_fd(rows,diff))) for lbl,rows in CR.items() if _fd(rows,diff)]
        rows_list += [(f"[NoRAG]{lbl}", _agg(_fd(rows,diff))) for lbl,rows in NR.items() if _fd(rows,diff)]
        rows_list.sort(key=lambda x: -x[1].get("bert_f1",0))
        for lbl,a in rows_list: _row(lbl, a)

    # TABLE 7 — Full Matrix
    _sec("TABLE 7 — Full Evaluation Matrix (All Questions)")
    _hdr()
    all_m = [(lbl,_agg(rows)) for lbl,rows in CR.items()]
    all_m += [(f"[NoRAG]{lbl}",_agg(rows)) for lbl,rows in NR.items()]
    all_m.sort(key=lambda x: -x[1].get("bert_f1",0))
    for lbl,a in all_m: _row(lbl, a)

    # TABLE 9 — RAG vs No-RAG
    if CR and NR:
        _sec("TABLE 9 — RAG vs No-RAG BERT-F1 Comparison")
        print(f"\n  {'Difficulty':<10} | {'RAG BERT-F1':<13} | {'No-RAG BERT-F1':<16} | Improvement")
        print(f"  {'-'*10}-+-{'-'*13}-+-{'-'*16}-+{'-'*12}")
        best_rag = max(CR.items(), key=lambda kv: _agg(kv[1]).get("bert_f1",0))
        best_nr  = max(NR.items(), key=lambda kv: _agg(kv[1]).get("bert_f1",0))
        for diff in diffs:
            rf = _fd(best_rag[1],diff); nf = _fd(best_nr[1],diff)
            if rf and nf:
                rb = _agg(rf)["bert_f1"]; nb = _agg(nf)["bert_f1"]
                print(f"  {diff.upper():<10} | {rb:<13.4f} | {nb:<16.4f} | +{rb-nb:.4f}")
        rb_all = _agg(best_rag[1])["bert_f1"]; nb_all = _agg(best_nr[1])["bert_f1"]
        print(f"  {'OVERALL':<10} | {rb_all:<13.4f} | {nb_all:<16.4f} | +{rb_all-nb_all:.4f}")
        print(f"\n  Best RAG  : {best_rag[0]}\n  No-RAG    : {best_nr[0]}")

    # LEADERBOARD
    _sec("LEADERBOARD — Ranked by BERT-F1")
    print(f"\n  # | {'Model/Combo':<{W}} | BF1    | R1     | RL     | METEOR | R@k    | Lat(ms)")
    print(f"  --+-{'-'*W}-+--------+--------+--------+--------+--------+--------")
    ranking = all_m if all_m else [(lbl,_agg(rows)) for lbl,rows in CR.items()]
    for i,(lbl,a) in enumerate(ranking,1):
        tag = " ◀ BEST" if i==1 else ""
        print(f"  {i:2} | {lbl[:W]:<{W}} | {a['bert_f1']:.4f} | {a['rouge1']:.4f} | "
              f"{a['rougeL']:.4f} | {a['meteor']:.4f} | {a['recall_at_k']:.4f} | {a['latency_ms']:7.1f}{tag}")

    # RECOMMENDATION
    if ranking:
        bl,ba = ranking[0]
        _sec("BEST MODEL RECOMMENDATION")
        print(f"\n  ✅ Best configuration: {bl}\n")
        for k,v in [("BERT-F1",ba['bert_f1']),("ROUGE-1",ba['rouge1']),
                    ("ROUGE-L",ba['rougeL']),("METEOR",ba['meteor']),
                    ("BLEU",ba['bleu']),("Recall@k",ba['recall_at_k']),
                    ("Latency",ba['latency_ms']),("Throughput",ba['throughput'])]:
            print(f"  {k:<14}: {v}")
        if CR and NR:
            rag_f = ba["bert_f1"]
            nr_f  = _agg(list(NR.values())[0])["bert_f1"]
            print(f"\n  RAG improvement over No-RAG: +{rag_f-nr_f:.4f} BERT-F1 "
                  f"({round((rag_f-nr_f)/max(nr_f,0.001)*100,1)}% relative)")

    print(f"\n{'═'*122}\n  ✅ Evaluation complete!\n{'═'*122}\n")


# ══════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════

def export_csv(output: Dict, csv_path: str):
    CR = output.get("combos",{}); NR = output.get("no_rag",{})
    diffs = ["easy","medium","hard","overall"]
    fields = ["model","type","difficulty","answered","total",
              "rouge1","rouge2","rougeL","meteor","bleu",
              "bert_p","bert_r","bert_f1","recall_at_k",
              "latency_ms","throughput","words_avg"]
    rows_out = []

    def add(lbl, rows, mtype):
        for diff in diffs:
            fr = rows if diff=="overall" else _fd(rows,diff)
            if not fr: continue
            a = _agg(fr)
            rows_out.append({"model":lbl,"type":mtype,"difficulty":diff,
                              "answered":int(a["answered"]),"total":int(a["total"]),
                              **{k:a.get(k,0.0) for k in fields[5:]}})

    for lbl,rows in CR.items(): add(lbl, rows, "RAG")
    for lbl,rows in NR.items(): add(lbl, rows, "NoRAG")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)
    print(f"  📊 CSV → {csv_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def run(pdf,gt_path,out_json,out_csv,combos,no_rag_models,mode):
    with open(gt_path,"r",encoding="utf-8") as f:
        gt = json.load(f)

    easy   = sum(1 for q in gt if q.get("difficulty")=="easy")
    medium = sum(1 for q in gt if q.get("difficulty")=="medium")
    hard   = sum(1 for q in gt if q.get("difficulty")=="hard")

    print(f"\n{'═'*70}\n  Kannada RAG Evaluation v2\n"
          f"  PDF:{pdf}  Q:{len(gt)} (E:{easy} M:{medium} H:{hard})\n"
          f"  Combos:{len(combos)}  Mode:{mode}\n  BERT:{_LOCAL_BERT or 'DISABLED'}\n{'═'*70}")

    CR = eval_combos(gt, pdf, combos)             if mode in ("all","combos_only","llm_compare") else {}
    NR = eval_no_rag(gt, no_rag_models)           if mode in ("all","no_rag_only") and no_rag_models else {}

    output = {"meta":{"pdf":pdf,"gt":gt_path,"timestamp":datetime.now().isoformat(),
                       "total":len(gt),"easy":easy,"medium":medium,"hard":hard,"mode":mode},
               "combos":CR, "no_rag":NR}

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
    print(f"\n  💾 JSON → {out_json}")

    export_csv(output, out_csv)
    generate_all_tables(output)
    return output


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    p  = argparse.ArgumentParser()
    p.add_argument("--pdf",         default="Kan-Science")
    p.add_argument("--gt",          default="evaluation/kannada_ground_truth_25.json")
    p.add_argument("--out",         default=f"evaluation/results_v2_{ts}.json")
    p.add_argument("--csv",         default=f"evaluation/results_v2_{ts}.csv")
    p.add_argument("--mode",        default="all",
                   choices=["all","combos_only","no_rag_only","llm_compare"])
    p.add_argument("--all_combos",  action="store_true")
    p.add_argument("--llm_compare", action="store_true")
    p.add_argument("--print_only",  default="")
    args = p.parse_args()

    if args.print_only:
        with open(args.print_only,"r",encoding="utf-8") as f:
            generate_all_tables(json.load(f))
        sys.exit(0)

    if args.mode == "llm_compare" or args.llm_compare:
        combos = LLM_COMPARISON_COMBOS
        print(f"  Mode: LLM comparison (KN-BERT hybrid × 4 LLMs)")
    elif args.all_combos:
        combos = ALL_COMBOS
        print(f"  Mode: All 8 combos")
    else:
        combos = [c for c in ALL_COMBOS if c.is_small_llm] or ALL_COMBOS[:4]
        print(f"  Mode: 8b-only combos ({len(combos)}/8)")

    nr_models = NO_RAG_MODELS if args.mode in ("all","no_rag_only") else []

    run(args.pdf, args.gt, args.out, args.csv,
        combos, nr_models, args.mode)