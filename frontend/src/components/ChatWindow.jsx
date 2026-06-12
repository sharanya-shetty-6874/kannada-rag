import React, { useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";

/**
 * Props:
 * - messages (array)
 * - playAudioForText (fn)
 * - isPlaying (bool)
 * - playingText (string)
 * - loading (bool)
 */
export default function ChatWindow({
  messages,
  playAudioForText,
  isPlaying,
  playingText,
  loading,
}) {
  const scrollRef = useRef();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  return (
    // ✅ FIX: added min-h-0 so this flex child can shrink and allow inner scroll
    <div className="flex-1 flex flex-col min-h-0">
      <header className="p-4 border-b dark:border-gray-700 bg-white dark:bg-gray-800 shrink-0">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            ಸಂಭಾಷಣೆ
          </h2>
          <div className="text-sm text-gray-500 dark:text-gray-300">ನಿಮ್ಮ PDF ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ</div>
        </div>
      </header>

      {/* ✅ FIX: flex-1 + min-h-0 + overflow-y-auto = proper scrollable area */}
      <main
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto p-4 bg-gray-50 dark:bg-gray-900"
      >
        <div className="max-w-3xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 dark:text-gray-400 py-8">
              ನೀವು ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ PDF ಬಗ್ಗೆ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ
            </div>
          )}

          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              msg={m}
              playAudio={() => playAudioForText(m.text)}
              isPlaying={isPlaying && playingText === m.text}
            />
          ))}

          {loading && (
            <div className="text-gray-500 italic text-center mt-4">ಉತ್ತರ ಬರಲಿದೆ...</div>
          )}
        </div>
      </main>
    </div>
  );
}