import os
import sys
import json
import argparse
import time

# ── IMPORTANT: set BEFORE any imports so config.py doesn't override ──────────
os.environ["HF_HUB_OFFLINE"]      = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.compare.compare_orchestrator import ALL_COMBOS, run_combo


# ── Error detection ───────────────────────────────────────────

_ERROR_PREFIXES = ("⚠️", "⏳")

def _is_error_answer(answer: str) -> bool:
    if not answer:
        return True
    return any(answer.strip().startswith(p) for p in _ERROR_PREFIXES)


# ── Sleep budgets ─────────────────────────────────────────────

_SLEEP_BETWEEN_QUESTIONS = {"small": 3.0, "large": 1.5}
_SLEEP_BETWEEN_COMBOS    = 8.0


# ── Force offline mode (restore for RAG pipeline) ────────────

def _force_offline():
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as _c
        _c.HF_HUB_OFFLINE = True
    except Exception:
        pass


# ── Locate the local kannada-bert snapshot ───────────────────

def _find_local_bertscore_model() -> str:
    main_models_dir = os.path.join(ROOT_DIR, "models")

    hub_cache_name = "models--l3cube-pune--kannada-sentence-bert-nli"
    hub_path = os.path.join(main_models_dir, hub_cache_name)
    if os.path.isdir(hub_path):
        snapshots_dir = os.path.join(hub_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snap in sorted(os.listdir(snapshots_dir)):
                snap_path = os.path.join(snapshots_dir, snap)
                if os.path.isfile(os.path.join(snap_path, "config.json")):
                    print(f"  🔍 BERTScore local model: {snap_path}")
                    return snap_path

    for candidate in [
        os.path.join(main_models_dir, "l3cube-pune", "kannada-sentence-bert-nli"),
        os.path.join(main_models_dir, "kannada-sentence-bert-nli"),
    ]:
        if os.path.isfile(os.path.join(candidate, "config.json")):
            print(f"  🔍 BERTScore local model: {candidate}")
            return candidate

    print("  ⚠️  Could not locate local kannada-bert model — BERTScore will be disabled.")
    return ""


_LOCAL_BERT_MODEL_PATH = _find_local_bertscore_model()


# ── Fully manual BERTScore (no bert_score library calls) ─────
#
# We do NOT use the bert_score library at all.
# Instead we:
#   1. Tokenize pred + ref with our local tokenizer
#   2. Run both through our local model to get token embeddings
#   3. L2-normalise each token embedding
#   4. Build a cosine similarity matrix between every token pair
#   5. Compute precision (max over ref for each pred token),
#      recall (max over pred for each ref token), and F1
#
# This is exactly what BERTScore does internally and produces
# equivalent scores, with zero network calls.
#
_BERTSCORE_AVAILABLE = True
_bs_tokenizer        = None
_bs_model            = None
_bs_device           = None


def _load_bertscore_model() -> bool:
    global _bs_tokenizer, _bs_model, _bs_device, _BERTSCORE_AVAILABLE

    if _bs_tokenizer is not None:
        return True

    if not _LOCAL_BERT_MODEL_PATH:
        _BERTSCORE_AVAILABLE = False
        return False

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        print("  📥 Loading BERTScore model (one-time)…")
        _bs_tokenizer = AutoTokenizer.from_pretrained(
            _LOCAL_BERT_MODEL_PATH, local_files_only=True
        )
        _bs_model = AutoModel.from_pretrained(
            _LOCAL_BERT_MODEL_PATH, local_files_only=True
        )
        _bs_device = "cuda" if torch.cuda.is_available() else "cpu"
        _bs_model.to(_bs_device)
        _bs_model.eval()
        print(f"  ✅ BERTScore model loaded on {_bs_device}.")
        return True
    except Exception as e:
        _BERTSCORE_AVAILABLE = False
        print(f"  ⚠️ BERTScore model load failed: {e}")
        print("     BERTScore disabled for this run.")
        return False


def _get_token_embeddings(text: str):
    """
    Tokenise `text` and return a 2-D tensor of shape (num_tokens, hidden_size)
    with L2-normalised embeddings (special tokens excluded).
    """
    import torch

    enc = _bs_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    enc = {k: v.to(_bs_device) for k, v in enc.items()}

    with torch.no_grad():
        out = _bs_model(**enc)

    # Use the last hidden state; squeeze the batch dimension
    hidden = out.last_hidden_state.squeeze(0)   # (seq_len, hidden)

    # Remove [CLS] and [SEP] tokens (first and last)
    hidden = hidden[1:-1]

    # L2 normalise each token vector so cosine sim == dot product
    norms  = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    hidden = hidden / norms

    return hidden   # (num_real_tokens, hidden_size)


def compute_bertscore(pred: str, ref: str) -> float:
    global _BERTSCORE_AVAILABLE

    if not _BERTSCORE_AVAILABLE:
        return 0.0

    if not _load_bertscore_model():
        return 0.0

    try:
        import torch

        emb_pred = _get_token_embeddings(pred)   # (P, H)
        emb_ref  = _get_token_embeddings(ref)    # (R, H)

        if emb_pred.shape[0] == 0 or emb_ref.shape[0] == 0:
            return 0.0

        # Cosine similarity matrix  (P x R)  — all values in [-1, 1]
        sim = torch.mm(emb_pred, emb_ref.T)

        # Precision: for each pred token, take best match in ref
        precision = sim.max(dim=1).values.mean().item()

        # Recall: for each ref token, take best match in pred
        recall    = sim.max(dim=0).values.mean().item()

        if precision + recall < 1e-9:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return round(float(f1), 4)

    except Exception as e:
        _BERTSCORE_AVAILABLE = False
        print(f"  ⚠️ BERTScore failed: {e}")
        print("     BERTScore disabled for this run.")
        return 0.0

    finally:
        _force_offline()   # always restore offline mode for RAG pipeline


# ── Other metric helpers ──────────────────────────────────────

def compute_rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result = scorer.score(target=ref, prediction=pred)
        return round(result["rougeL"].fmeasure, 4)
    except Exception as e:
        print(f"  ⚠️ ROUGE-L error: {e}")
        return 0.0


def compute_bleu(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        return round(sentence_bleu(
            [list(ref)], list(pred), smoothing_function=smoother
        ), 4)
    except Exception as e:
        print(f"  ⚠️ BLEU error: {e}")
        return 0.0


def compute_exact_match(pred: str, ref: str) -> int:
    return int(pred.strip() == ref.strip())


# ── Main evaluation loop ──────────────────────────────────────

def run_evaluation(pdf_name: str, gt_path: str, output_path: str, combos_override=None):
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    active_combos = combos_override if combos_override is not None else ALL_COMBOS
    total_calls   = len(active_combos) * len(ground_truth)

    print(f"\n📋 Ground truth questions : {len(ground_truth)}")
    print(f"🔧 Active combos          : {len(active_combos)}/8")
    print(f"📘 PDF                    : {pdf_name}")
    print(f"🔢 Total LLM calls        : {len(active_combos)} combos × {len(ground_truth)} questions = {total_calls}")
    print(f"⏱️  Approx time            : ~{total_calls * 2 // 60}–{total_calls * 4 // 60} min (free tier)")
    print(f"\n⚠️  Rate-limited / error answers will be SKIPPED from metrics but logged.\n")
    print(f"🤖 BERTScore model        : {_LOCAL_BERT_MODEL_PATH or 'NOT FOUND — disabled'}\n")

    results     = {}
    call_number = 0

    for combo_idx, combo in enumerate(active_combos):
        label      = combo.label
        sleep_secs = _SLEEP_BETWEEN_QUESTIONS["small" if combo.is_small_llm else "large"]

        print(f"\n{'='*62}")
        print(f"🔧 COMBO {combo_idx+1}/{len(active_combos)}: {label}")
        print(f"{'='*62}")

        combo_results = {
            "config":       combo.to_dict(),
            "per_question": {},
            "summary":      {},
        }

        rouge_scores, bert_scores, bleu_scores = [], [], []
        conf_scores, latency_list, kn_ratio_list = [], [], []
        hall_counts   = {"low": 0, "medium": 0, "high": 0}
        skipped_count = 0

        for q_idx, item in enumerate(ground_truth):
            qid      = item["id"]
            question = item["question"]
            ref_ans  = item["ground_truth_answer"]
            call_number += 1

            print(f"  [{call_number:>3}/{total_calls}] [{qid}] {question[:55]}...")

            result   = run_combo(question, pdf_name, combo)
            pred_ans = result.get("answer", "")

            if _is_error_answer(pred_ans):
                skipped_count += 1
                print(f"         ⚠️  SKIPPED: {pred_ans[:80]}")
                combo_results["per_question"][qid] = {
                    "question":         question,
                    "predicted_answer": pred_ans,
                    "reference_answer": ref_ans,
                    "skipped":          True,
                    "skip_reason":      pred_ans[:120],
                    "latency_ms":       result.get("latency_ms", 0),
                }
                time.sleep(sleep_secs)
                continue

            rouge_l = compute_rouge_l(pred_ans, ref_ans)
            bert_f1 = compute_bertscore(pred_ans, ref_ans)
            bleu    = compute_bleu(pred_ans, ref_ans)
            exact   = compute_exact_match(pred_ans, ref_ans)
            conf    = result.get("confidence", 0.0)
            kn_r    = result.get("kannada_ratio", 0.0)
            lat     = result.get("latency_ms", 0.0)
            hall    = result.get("hallucination_risk", "low")

            rouge_scores.append(rouge_l)
            bert_scores.append(bert_f1)
            bleu_scores.append(bleu)
            conf_scores.append(conf)
            latency_list.append(lat)
            kn_ratio_list.append(kn_r)
            hall_counts[hall] = hall_counts.get(hall, 0) + 1

            print(f"         ✅ ROUGE-L={rouge_l:.3f}  BERT={bert_f1:.3f}  BLEU={bleu:.3f}  "
                  f"Conf={conf:.2f}  Lat={lat:.0f}ms")
            print(f"         💬 {pred_ans[:80]}...")

            combo_results["per_question"][qid] = {
                "question":           question,
                "predicted_answer":   pred_ans,
                "reference_answer":   ref_ans,
                "skipped":            False,
                "rougeL":             rouge_l,
                "bertscore_f1":       bert_f1,
                "bleu":               bleu,
                "exact_match":        exact,
                "confidence":         conf,
                "kannada_ratio":      kn_r,
                "hallucination_risk": hall,
                "chunks_used":        result.get("chunks_used", 0),
                "latency_ms":         lat,
                "top_chunk_score":    result.get("top_chunk_score", 0.0),
            }

            if q_idx < len(ground_truth) - 1:
                time.sleep(sleep_secs)

        def _avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

        answered = len(ground_truth) - skipped_count
        combo_results["summary"] = {
            "rougeL_mean":                    _avg(rouge_scores),
            "bertscore_f1_mean":              _avg(bert_scores),
            "bleu_mean":                      _avg(bleu_scores),
            "confidence_mean":                _avg(conf_scores),
            "kannada_ratio_mean":             _avg(kn_ratio_list),
            "latency_ms_mean":                _avg(latency_list),
            "hallucination_risk_distribution": hall_counts,
            "total_questions":                len(ground_truth),
            "answered_questions":             answered,
            "skipped_questions":              skipped_count,
        }

        results[label] = combo_results

        print(f"\n  📊 Answered {answered}/{len(ground_truth)} | "
              f"ROUGE-L={_avg(rouge_scores):.3f}  BERTScore={_avg(bert_scores):.3f}  "
              f"BLEU={_avg(bleu_scores):.3f}  Lat={_avg(latency_list):.0f}ms")
        if skipped_count:
            print(f"  ⚠️  {skipped_count} skipped due to rate limits / errors")

        if combo_idx < len(active_combos) - 1:
            print(f"\n  💤 Pausing {_SLEEP_BETWEEN_COMBOS:.0f}s before next combo...")
            time.sleep(_SLEEP_BETWEEN_COMBOS)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Evaluation complete! Saved to: {output_path}")
    _print_leaderboard(results)


# ── Leaderboard ───────────────────────────────────────────────

def _print_leaderboard(results: dict):
    print("\n" + "="*95)
    print("🏆  LEADERBOARD — sorted by BERTScore F1")
    print("="*95)
    rows = []
    for label, data in results.items():
        s = data["summary"]
        rows.append({
            "label":      label,
            "rougeL":     s["rougeL_mean"],
            "bertscore":  s["bertscore_f1_mean"],
            "bleu":       s["bleu_mean"],
            "confidence": s["confidence_mean"],
            "latency":    s["latency_ms_mean"],
            "answered":   f"{s.get('answered_questions', 0)}/{s.get('total_questions', 0)}",
        })
    rows.sort(key=lambda x: -x["bertscore"])
    print(f"{'#':<3} {'Combo':<45} {'Ans':<6} {'ROUGE-L':<9} {'BERTScore':<11} {'BLEU':<8} {'Conf':<7} {'Lat(ms)'}")
    print("-"*100)
    for i, r in enumerate(rows, 1):
        print(f"{i:<3} {r['label']:<45} {r['answered']:<6} {r['rougeL']:<9.3f} "
              f"{r['bertscore']:<11.3f} {r['bleu']:<8.3f} {r['confidence']:<7.3f} {r['latency']:.0f}")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime
    from src.compare.compare_orchestrator import ALL_COMBOS as _ALL

    _ts = datetime.now().strftime("%Y%m%d_%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--gt",  required=True)
    parser.add_argument("--out", default=f"evaluation/results_{_ts}.json")
    parser.add_argument("--all", dest="run_all", action="store_true", default=False,
                        help="Run all 8 combos (default: 8b only)")
    args = parser.parse_args()

    if args.run_all:
        active_combos = _ALL
        print("🔢 Running ALL 8 combos (4× 8b + 4× 70b)")
    else:
        active_combos = [c for c in _ALL if "8b" in c.llm_model]
        print(f"🔢 Running 8b-only combos ({len(active_combos)}/8)  — use --all for all 8")

    print(f"💾 Results will be saved to: {args.out}")
    run_evaluation(args.pdf, args.gt, args.out, combos_override=active_combos)