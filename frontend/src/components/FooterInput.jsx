// src/components/FooterInput.jsx
import React, { useState, useRef, useEffect } from "react";
import Keyboard from "react-simple-keyboard";
import "react-simple-keyboard/build/css/index.css";

import phoneticKannada, { applyMatra, autoHalant } from "../kannadaTypingEngine";
import { kannadaKeyboardLayouts } from "../kannadaKeyboard";
import { longPressKeys } from "../longPressOptions";

export default function FooterInput({
  question,
  setQuestion,
  askQuestion,
  showKeyboard,
  setShowKeyboard,
  playAudioForText,
  isPlaying,
  darkMode
}){

  const [keyboardLayer, setKeyboardLayer] = useState("default"); // default, english, numbers, symbols, emoji
  const [popupOptions, setPopupOptions] = useState([]);
  const [popupVisible, setPopupVisible] = useState(false);

  const keyboardRef = useRef(null);
  const containerRef = useRef(null);

  // Close popup when clicking outside
  useEffect(() => {
    const close = (e) => {
      // if click happened inside popup or keyboard, ignore
      if (!containerRef.current) return;
      if (containerRef.current.contains(e.target)) return;
      setPopupOptions([]);
      setPopupVisible(false);
    };
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, []);

  // handle keys coming from virtual keyboard
  const onKeyPress = (button) => {
    // layer keys
    if (button === "{numbers}" || button === "{symbols}" || button === "{emoji}" || button === "{english}" || button === "{default}") {
      const name = button.replace(/[{}]/g, "");
      setKeyboardLayer(name);
      // clear any popup when switching
      setPopupOptions([]);
      setPopupVisible(false);
      return;
    }

    // backspace
    if (button === "{bksp}") {
      // remove last grapheme properly
      setQuestion((q) => q.slice(0, -1));
      return;
    }

    // space behaviour: on english layer confirm transliteration then add space
    if (button === "{space}") {
      if (keyboardLayer === "english") {
        // confirm transliteration of trailing token, then add space
        setQuestion((q) => {
          const translit = phoneticKannada(q);
          return translit + " ";
        });
        return;
      } else {
        setQuestion((q) => q + " ");
        return;
      }
    }

    // handle long-press popup activation (only on Kannada layer)
    if (keyboardLayer !== "english" && longPressKeys[button]) {
      setPopupOptions(longPressKeys[button]);
      setPopupVisible(true);
      return;
    }

    // Normal typing:
    // For english layer -> live phonetic transliteration (replace trailing token)
    // For other layers -> place raw Kannada char and apply matra/halant rules
    if (keyboardLayer === "english") {
      // append the latin char and transliterate trailing token live
      setQuestion((q) => {
        const updated = q + button;
        return phoneticKannada(updated);
      });
    } else {
      // non-english: add the literal button (which is a Kannada glyph or matra)
      setQuestion((q) => {
        let updated = q + button;
        updated = applyMatra(updated);
        updated = autoHalant(updated);
        return updated;
      });
    }
  };

  // clicking an option from popup
  const selectPopupOption = (opt) => {
    setQuestion((q) => {
      // if last char is a base consonant in string, replace it with combined form,
      // but simpler: append option (most options are full forms)
      return q + opt;
    });
    setPopupOptions([]);
    setPopupVisible(false);
  };

  return (
    <div
      ref={containerRef}
      className={`p-4 border-t ${darkMode ? "bg-gray-900 border-gray-700" : "bg-white border-gray-200"}`}
    >
      <div className="max-w-5xl mx-auto flex gap-2 items-center">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಬರೆಯಿರಿ..."
          className={`flex-1 p-3 rounded-lg border ${darkMode ? "bg-gray-800 text-white border-gray-700" : "bg-gray-100 text-black border-gray-200"}`}
        />
       

        <button
          onClick={() => askQuestion({})}
          className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          ಕಳುಹಿಸಿ
        </button>

        <button
          onClick={() => {
            // replay last bot or play last text
            playAudioForText();
          }}
          className={`px-4 py-2 rounded-lg font-semibold ${isPlaying ? "bg-red-600 text-white" : "bg-yellow-400 text-black"}`}
        >
          {isPlaying ? "⏸" : "🔊"}
        </button>

        <button
          onClick={() => setShowKeyboard((s) => !s)}
          className="px-3 py-2 rounded-lg bg-gray-300 dark:bg-gray-700 text-black dark:text-white"
        >
          {showKeyboard ? "⌨️ ಮರೆಮಾಡಿ" : "⌨️ ತೋರಿಸಿ"}
        </button>
      </div>

      {/* keyboard area */}
      {showKeyboard && (
        <div className="mt-3">
          {/* Popup (centered above keyboard) */}
          {popupVisible && popupOptions.length > 0 && (
            <div className="keyboard-popup-wrapper flex justify-center mb-2">
              <div className="keyboard-popup bg-white dark:bg-gray-800 border rounded-lg shadow-md px-2 py-1 flex gap-2">
                {popupOptions.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => selectPopupOption(opt)}
                    className="px-3 py-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="keyboard-container">
            <Keyboard
              keyboardRef={(r) => (keyboardRef.current = r)}
              layout={kannadaKeyboardLayouts}
              layoutName={keyboardLayer}
              display={{
                "{bksp}": "⌫",
                "{space}": "Space",
                "{numbers}": "123",
                "{symbols}": "#+=",
                "{emoji}": "😊",
                "{english}": "ಇಂಗ್ಲಿಷ್",
                "{default}": "ಕನ್ನಡ"
              }}
              theme={"hg-theme-default custom-kbd"}
              onKeyPress={onKeyPress}
            />
          </div>
        </div>
      )}
    </div>
  );
}
