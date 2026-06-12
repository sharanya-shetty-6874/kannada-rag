import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";
import FooterInput from "./components/FooterInput";
import ComparePanel from "./components/ComparePanel";
import "./index.css";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("theme") === "dark";
  });

  const [chats, setChats] = useState(() => {
    try {
      const raw = localStorage.getItem("chats_v1");
      return raw ? JSON.parse(raw) : {};
    } catch {
      return {};
    }
  });

  const [activeChatId, setActiveChatId] = useState(() => {
    return localStorage.getItem("activeChatId") || "";
  });

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [showKeyboard, setShowKeyboard] = useState(false);
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "compare"

  const [currentPDF, setCurrentPDF] = useState(() => {
    return localStorage.getItem("currentPDF") || "";
  });

  // ── Agent metadata for last query ─────────────────────────
  const [agentMeta, setAgentMeta] = useState(null);

  const audioCacheRef   = useRef({});
  const currentAudioRef = useRef(null);
  const [isPlaying, setIsPlaying]   = useState(false);
  const [playingText, setPlayingText] = useState("");

  // Ensure at least one chat exists
  useEffect(() => {
    if (Object.keys(chats).length === 0) {
      const id = Date.now().toString();
      setChats({ [id]: { title: "New Chat", messages: [], pdf: "" } });
      setActiveChatId(id);
    } else if (!activeChatId) {
      setActiveChatId(Object.keys(chats)[0]);
    }
  }, []);

  useEffect(() => { localStorage.setItem("chats_v1", JSON.stringify(chats)); }, [chats]);
  useEffect(() => { if (activeChatId) localStorage.setItem("activeChatId", activeChatId); }, [activeChatId]);
  useEffect(() => { localStorage.setItem("currentPDF", currentPDF || ""); }, [currentPDF]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const createNewChat = () => {
    const id = Date.now().toString();
    setChats((prev) => ({
      ...prev,
      [id]: { title: "New Chat", messages: [], pdf: currentPDF },
    }));
    setActiveChatId(id);
    setAgentMeta(null);
  };

  const deleteChat = (id) => {
    setChats((prev) => {
      const copy = { ...prev };
      delete copy[id];
      const remaining = Object.keys(copy);
      setActiveChatId(remaining[0] || "");
      return copy;
    });
  };

  const appendMessageToActiveChat = (msg) => {
    if (!activeChatId) return;
    setChats((prev) => {
      const chat = prev[activeChatId];
      if (!chat) return prev;
      return { ...prev, [activeChatId]: { ...chat, messages: [...chat.messages, msg] } };
    });
  };

  const updateActiveChatMeta = (meta) => {
    if (!activeChatId) return;
    setChats((prev) => {
      const chat = prev[activeChatId];
      if (!chat) return prev;
      return { ...prev, [activeChatId]: { ...chat, ...meta } };
    });
  };

  // AUDIO
  const playAudioForText = async (text) => {
    if (!text) return;
    try {
      if (isPlaying && playingText === text) {
        if (currentAudioRef.current) { currentAudioRef.current.pause(); currentAudioRef.current.currentTime = 0; }
        setIsPlaying(false); setPlayingText(""); return;
      }
      if (currentAudioRef.current) { currentAudioRef.current.pause(); currentAudioRef.current.currentTime = 0; }
      setIsPlaying(false); setPlayingText("");
      let url = audioCacheRef.current[text];
      if (!url) {
        const res = await axios.post(`${API_BASE}/tts`, { text });
        url = res.data.url;
        audioCacheRef.current[text] = url;
      }
      const audio = new Audio(url);
      currentAudioRef.current = audio;
      setPlayingText(text); setIsPlaying(true);
      audio.play().catch(() => setIsPlaying(false));
      audio.onended = () => { setIsPlaying(false); setPlayingText(""); currentAudioRef.current = null; };
    } catch {
      setIsPlaying(false); setPlayingText("");
    }
  };

  // ── Poll for PDF ready status ─────────────────────────────
  const pollPdfStatus = (base) => {
    return new Promise((resolve) => {
      const interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE}/upload_status/${base}`);
          const status = res.data.status;
          if (status === "ready") { clearInterval(interval); resolve(true); }
          else if (status === "error") { clearInterval(interval); resolve(false); }
        } catch { clearInterval(interval); resolve(false); }
      }, 2000);
    });
  };

  // ── Ask question ──────────────────────────────────────────
  const askQuestion = async ({ question: q, pdfOverride } = {}) => {
    const text = q ?? question;
    if (!text.trim()) return;

    if (!currentPDF && !pdfOverride) {
      alert("ದಯವಿಟ್ಟು ಮೊದಲು PDF upload ಮಾಡಿ.");
      return;
    }

    const pdfToUse = pdfOverride || currentPDF;
    const existingBotCount =
      chats[activeChatId]?.messages.filter((m) => m.role === "bot").length || 0;

    appendMessageToActiveChat({ role: "user", text });
    setLoading(true);
    setQuestion("");
    setAgentMeta(null);

    try {
      const res = await axios.post(`${API_BASE}/query`, {
        question: text,
        pdf: pdfToUse,
      });

      const data = res.data;

      const ans =
        typeof data.answer === "string"
          ? data.answer
          : "ಕ್ಷಮಿಸಿ, ಮಾಹಿತಿ ದೊರೆಯಲಿಲ್ಲ.";

      // ── Store agent metadata ──────────────────────────────
      setAgentMeta({
        intent:            data.intent || "",
        sub_intents:       data.sub_intents || [],
        entities:          data.entities || [],
        confidence:        data.confidence ?? null,
        hallucination_risk: data.hallucination_risk || "low",
        quality:           data.quality || "",
        chunks_used:       data.chunks_used ?? 0,
        domain:            data.domain || "",
        retry_count:       data.retry_count ?? 0,
        latency_ms:        data.latency_ms ?? 0,
      });

      appendMessageToActiveChat({ role: "bot", text: ans });

      if (existingBotCount === 0) {
        const title = ans.slice(0, 40).replace(/\n/g, " ") + (ans.length > 40 ? "..." : "");
        updateActiveChatMeta({ title });
      }
    } catch (err) {
      if (err.response?.status === 202) {
        appendMessageToActiveChat({
          role: "bot",
          text: "⏳ PDF ಇನ್ನೂ index ಆಗುತ್ತಿದೆ. ಸ್ವಲ್ಪ ಕಾಯಿರಿ...",
        });
      } else {
        appendMessageToActiveChat({
          role: "bot",
          text: "ಸರ್ವರ್‌ನಲ್ಲಿ ತಪ್ಪು — ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  // ── Upload PDF (async — polls until ready) ────────────────
  const uploadPDF = async (file) => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    const base = file.name.replace(/\.pdf$/i, "");

    try {
      await axios.post(`${API_BASE}/upload_pdf`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setCurrentPDF(base);
      updateActiveChatMeta({ pdf: base });

      alert("⏳ PDF upload accepted! Indexing in background...\nYou'll be able to ask questions once it's ready.");

      pollPdfStatus(base).then((success) => {
        if (success) {
          alert(`✅ "${base}" is ready! You can now ask questions.`);
        } else {
          alert(`❌ Indexing failed for "${base}". Please re-upload.`);
        }
      });

    } catch {
      alert("PDF upload failed.");
    }
  };

  return (
    // ✅ FIX: h-screen + overflow-hidden is correct for full-screen app layout
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        setActiveChatId={setActiveChatId}
        createNewChat={createNewChat}
        deleteChat={deleteChat}
        currentPDF={currentPDF}
        uploadPDF={uploadPDF}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
      />

      {/* ── RIGHT PANEL: full height flex column ─────────── */}
      <div className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-900 min-h-0 overflow-hidden">

        {/* ── TAB BAR (fixed, never scrolls) ──────────────── */}
        <div className="flex border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shrink-0">
          <button
            onClick={() => setActiveTab("chat")}
            className={`px-6 py-3 text-sm font-semibold transition-colors border-b-2
              ${activeTab === "chat"
                ? "border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
          >
            💬 Chat
          </button>
          <button
            onClick={() => setActiveTab("compare")}
            className={`px-6 py-3 text-sm font-semibold transition-colors border-b-2
              ${activeTab === "compare"
                ? "border-blue-500 text-blue-600 dark:text-blue-400"
                : "border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"}`}
          >
            🔬 Compare Pipelines
          </button>
        </div>

        {/* ── CHAT TAB ─────────────────────────────────────── */}
        {activeTab === "chat" && (
          // ✅ FIX: flex-1 + min-h-0 + flex-col lets ChatWindow scroll properly
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
            {/* Agent metadata bar (fixed, never scrolls) */}
            {agentMeta && (
              <div className="agent-meta-bar px-4 py-2 text-xs flex flex-wrap gap-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shrink-0">
                <MetaBadge label="Intent"      value={agentMeta.intent} />
                {agentMeta.sub_intents?.length > 0 && (
                  <MetaBadge label="Sub-intents" value={agentMeta.sub_intents.join(", ")} />
                )}
                {agentMeta.entities?.length > 0 && (
                  <MetaBadge label="Entities" value={agentMeta.entities.slice(0,4).join(", ")} />
                )}
                <MetaBadge
                  label="Confidence"
                  value={agentMeta.confidence !== null ? `${(agentMeta.confidence * 100).toFixed(0)}%` : "—"}
                  color={agentMeta.confidence >= 0.6 ? "green" : agentMeta.confidence >= 0.35 ? "yellow" : "red"}
                />
                <MetaBadge
                  label="Hallucination"
                  value={agentMeta.hallucination_risk}
                  color={agentMeta.hallucination_risk === "low" ? "green" : agentMeta.hallucination_risk === "medium" ? "yellow" : "red"}
                />
                <MetaBadge label="Domain"     value={agentMeta.domain} />
                <MetaBadge label="Chunks"     value={agentMeta.chunks_used} />
                {agentMeta.retry_count > 0 && (
                  <MetaBadge label="Retries" value={agentMeta.retry_count} color="yellow" />
                )}
                <MetaBadge label="Latency" value={`${agentMeta.latency_ms}ms`} />
              </div>
            )}

            {/* ✅ ChatWindow now gets a properly bounded flex container */}
            <ChatWindow
              messages={chats[activeChatId]?.messages || []}
              playAudioForText={playAudioForText}
              isPlaying={isPlaying}
              playingText={playingText}
              loading={loading}
            />

            {/* Footer is shrink-0 so it never gets squished */}
            <FooterInput
              question={question}
              setQuestion={setQuestion}
              askQuestion={askQuestion}
              showKeyboard={showKeyboard}
              setShowKeyboard={setShowKeyboard}
              playAudioForText={() => {
                const msgs = chats[activeChatId]?.messages || [];
                const lastBot = [...msgs].reverse().find((m) => m.role === "bot");
                if (lastBot) playAudioForText(lastBot.text);
              }}
              isPlaying={isPlaying}
              darkMode={darkMode}
            />
          </div>
        )}

        {/* ── COMPARE TAB ──────────────────────────────────── */}
        {activeTab === "compare" && (
          <div className="flex-1 overflow-y-auto min-h-0">
            <ComparePanel
              currentPDF={currentPDF}
              darkMode={darkMode}
            />
          </div>
        )}

      </div>
    </div>
  );
}

// ── Agent metadata badge component ───────────────────────────
function MetaBadge({ label, value, color }) {
  const colorClass =
    color === "green"  ? "text-green-600 dark:text-green-400" :
    color === "yellow" ? "text-yellow-600 dark:text-yellow-400" :
    color === "red"    ? "text-red-600 dark:text-red-400" :
    "text-gray-500 dark:text-gray-400";

  return (
    <span className="flex items-center gap-1">
      <span className="text-gray-400 dark:text-gray-500">{label}:</span>
      <span className={`font-medium ${colorClass}`}>{String(value)}</span>
    </span>
  );
}