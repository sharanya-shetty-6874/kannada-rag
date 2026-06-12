/**
 * src/components/ComparePanel.jsx
 *
 * Comparison UI — lets user pick combos, run them, and see
 * a results table with Answer, Confidence, Hallucination Risk,
 * Latency, Chunks Used, Kannada Ratio side by side.
 *
 * Props:
 *   currentPDF   — string: current pdf base name
 *   darkMode     — bool
 */

import React, { useState } from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

// ── All 8 combo definitions (mirrors backend ALL_COMBOS) ──────
const ALL_COMBOS = [
  { embed_model: "kannada-bert", retrieval: "hybrid", llm_model: "llama-3.3-70b-versatile" },
  { embed_model: "kannada-bert", retrieval: "hybrid", llm_model: "llama-3.1-8b-instant"    },
  { embed_model: "kannada-bert", retrieval: "dense",  llm_model: "llama-3.3-70b-versatile" },
  { embed_model: "kannada-bert", retrieval: "dense",  llm_model: "llama-3.1-8b-instant"    },
  { embed_model: "e5",           retrieval: "hybrid", llm_model: "llama-3.3-70b-versatile" },
  { embed_model: "e5",           retrieval: "hybrid", llm_model: "llama-3.1-8b-instant"    },
  { embed_model: "e5",           retrieval: "dense",  llm_model: "llama-3.3-70b-versatile" },
  { embed_model: "e5",           retrieval: "dense",  llm_model: "llama-3.1-8b-instant"    },
];

function comboLabel(c) {
  const emb = c.embed_model === "kannada-bert" ? "KN-BERT" : "E5";
  const ret = c.retrieval === "hybrid" ? "Hybrid" : "Dense";
  const llm = c.llm_model?.includes("3.3") ? "Llama-70B" : "Llama-8B";
  return `${emb} · ${ret} · ${llm}`;
}

// ── Risk badge ────────────────────────────────────────────────
function RiskBadge({ risk }) {
  const styles = {
    low:    "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
    medium: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
    high:   "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${styles[risk] || styles.low}`}>
      {risk}
    </span>
  );
}

// ── Confidence bar ────────────────────────────────────────────
function ConfBar({ value }) {
  const pct   = Math.round((value || 0) * 100);
  const color = pct >= 60 ? "bg-green-500" : pct >= 35 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div className={`${color} h-2 rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs w-8 text-right">{pct}%</span>
    </div>
  );
}

// ── Main ComparePanel ─────────────────────────────────────────
export default function ComparePanel({ currentPDF, darkMode }) {
  const [question,    setQuestion]    = useState("");
  const [selected,    setSelected]    = useState(ALL_COMBOS.map((_, i) => i));
  const [results,     setResults]     = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState("");
  const [expandedIdx, setExpandedIdx] = useState(null);

  const toggleCombo = (i) => {
    setSelected((prev) =>
      prev.includes(i) ? prev.filter((x) => x !== i) : [...prev, i]
    );
  };

  const runCompare = async () => {
    if (!question.trim()) { setError("ಪ್ರಶ್ನೆ ನಮೂದಿಸಿ"); return; }
    if (!currentPDF)       { setError("ಮೊದಲು PDF upload ಮಾಡಿ"); return; }
    if (selected.length === 0) { setError("ಕನಿಷ್ಠ ಒಂದು combo ಆಯ್ಕೆ ಮಾಡಿ"); return; }

    setError("");
    setLoading(true);
    setResults([]);
    setExpandedIdx(null);

    const combos = selected.map((i) => ALL_COMBOS[i]);

    try {
      const res = await axios.post(`${API_BASE}/compare`, {
        question,
        pdf: currentPDF,
        combos,
      });
      setResults(res.data.results || []);
    } catch (err) {
      setError("Server error: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Sort results by confidence desc for display
  const sortedResults = [...results].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

  return (
    <div className="flex flex-col w-full p-4 gap-4 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">

      {/* Header */}
      <div>
        <h2 className="text-xl font-bold mb-1">🔬 Pipeline Comparison</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Compare 8 combos: 2 embedding models × 2 retrieval modes × 2 LLMs
        </p>
      </div>

      {/* Combo selector */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <span className="font-semibold text-sm">Select Combos</span>
          <div className="flex gap-2">
            <button
              onClick={() => setSelected(ALL_COMBOS.map((_, i) => i))}
              className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
            >
              All
            </button>
            <button
              onClick={() => setSelected([])}
              className="text-xs text-gray-500 hover:underline"
            >
              None
            </button>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {ALL_COMBOS.map((combo, i) => (
            <label
              key={i}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors text-sm
                ${selected.includes(i)
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30"
                  : "border-gray-200 dark:border-gray-700"}`}
            >
              <input
                type="checkbox"
                checked={selected.includes(i)}
                onChange={() => toggleCombo(i)}
                className="accent-blue-500"
              />
              <span className="font-mono">{comboLabel(combo)}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Question input */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm">
        <label className="block text-sm font-semibold mb-2">ಪ್ರಶ್ನೆ ನಮೂದಿಸಿ</label>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          placeholder="ಆಹಾರ ವೈವಿಧ್ಯ ಎಂದರೆ ಏನು?"
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-sm
                     bg-white dark:bg-gray-900 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {currentPDF && (
          <p className="mt-1 text-xs text-gray-400">PDF: {currentPDF}</p>
        )}
        {error && (
          <p className="mt-1 text-xs text-red-500">{error}</p>
        )}
        <button
          onClick={runCompare}
          disabled={loading}
          className="mt-3 w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white
                     font-semibold text-sm py-2 px-4 rounded-lg transition-colors"
        >
          {loading
            ? `⏳ Running ${selected.length} combo${selected.length !== 1 ? "s" : ""} sequentially…`
            : `▶ Run ${selected.length} Combo${selected.length !== 1 ? "s" : ""}`}
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm flex flex-col items-center gap-3">
          <div className="animate-spin text-3xl">⚙️</div>
          <p className="text-sm text-gray-500">
            Running {selected.length} pipeline configuration{selected.length !== 1 ? "s" : ""} sequentially…
          </p>
          <p className="text-xs text-gray-400">
            Each combo runs one after the other to avoid model loading conflicts.
            Expected time: {selected.length * 25}–{selected.length * 45} seconds.
          </p>
        </div>
      )}

      {/* Results table */}
      {sortedResults.length > 0 && !loading && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 flex items-center justify-between">
            <span className="font-semibold text-sm">Results — {sortedResults.length} combos</span>
            <span className="text-xs text-gray-400">Sorted by Confidence ↓</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 dark:bg-gray-900 text-left text-xs text-gray-500 uppercase tracking-wide">
                  <th className="px-4 py-3 font-semibold">#</th>
                  <th className="px-4 py-3 font-semibold">Combo</th>
                  <th className="px-4 py-3 font-semibold">Confidence</th>
                  <th className="px-4 py-3 font-semibold">Hallucination</th>
                  <th className="px-4 py-3 font-semibold">Chunks</th>
                  <th className="px-4 py-3 font-semibold">KN Ratio</th>
                  <th className="px-4 py-3 font-semibold">Latency</th>
                  <th className="px-4 py-3 font-semibold">Answer</th>
                </tr>
              </thead>
              <tbody>
                {sortedResults.map((r, i) => {
                  const cfg      = r.config || {};
                  const label    = comboLabel(cfg);
                  const isExpanded = expandedIdx === i;

                  return (
                    <React.Fragment key={i}>
                      <tr
                        className={`border-t border-gray-100 dark:border-gray-700 hover:bg-gray-50
                                    dark:hover:bg-gray-700 cursor-pointer transition-colors
                                    ${i === 0 ? "bg-green-50 dark:bg-green-900/10" : ""}`}
                        onClick={() => setExpandedIdx(isExpanded ? null : i)}
                      >
                        <td className="px-4 py-3 font-bold text-gray-400">
                          {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}
                        </td>
                        <td className="px-4 py-3">
                          <div className="font-mono text-xs leading-5">
                            <div className="font-semibold text-blue-600 dark:text-blue-400">
                              {cfg.embed_model === "kannada-bert" ? "KN-BERT" : "E5"}
                            </div>
                            <div className="text-gray-500">
                              {cfg.retrieval === "hybrid" ? "🔀 Hybrid" : "🎯 Dense"}
                            </div>
                            <div className="text-gray-500">
                              {cfg.llm_model?.includes("3.3") ? "🦙 Llama-70B" : "🦙 Llama-8B"}
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 min-w-[120px]">
                          <ConfBar value={r.confidence} />
                        </td>
                        <td className="px-4 py-3">
                          <RiskBadge risk={r.hallucination_risk || "low"} />
                        </td>
                        <td className="px-4 py-3 text-center font-mono">
                          {r.chunks_used ?? "—"}
                        </td>
                        <td className="px-4 py-3 text-center font-mono">
                          {r.kannada_ratio != null
                            ? `${(r.kannada_ratio * 100).toFixed(0)}%`
                            : "—"}
                        </td>
                        <td className="px-4 py-3 font-mono text-xs">
                          {r.latency_ms != null ? `${(r.latency_ms / 1000).toFixed(1)}s` : "—"}
                        </td>
                        <td className="px-4 py-3 max-w-[220px]">
                          <p className="text-xs text-gray-700 dark:text-gray-300 line-clamp-2">
                            {r.answer || "—"}
                          </p>
                          <button className="text-xs text-blue-500 mt-1 hover:underline">
                            {isExpanded ? "▲ ಮರೆಮಾಡಿ" : "▼ ಸಂಪೂರ್ಣ ನೋಡಿ"}
                          </button>
                        </td>
                      </tr>

                      {/* Expanded answer row */}
                      {isExpanded && (
                        <tr className="bg-blue-50 dark:bg-blue-900/10">
                          <td colSpan={8} className="px-6 py-4">
                            <div className="font-semibold text-xs text-gray-500 mb-1 uppercase tracking-wide">
                              Full Answer — {label}
                            </div>
                            <p className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap">
                              {r.answer}
                            </p>
                            {r.error && (
                              <p className="mt-2 text-xs text-red-500">Error: {r.error}</p>
                            )}
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div className="px-4 py-3 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-400 flex flex-wrap gap-4">
            <span>🥇 Best confidence</span>
            <span>KN-BERT = kannada-sentence-bert-nli (768-d)</span>
            <span>E5 = multilingual-e5-small (384-d)</span>
            <span>🔀 Hybrid = FAISS + BM25</span>
            <span>🎯 Dense = FAISS only</span>
            <span>🦙 Llama-70B = llama-3.3-70b-versatile</span>
            <span>🦙 Llama-8B = llama-3.1-8b-instant</span>
          </div>
        </div>
      )}

    </div>
  );
}