import React from "react";

/**
 * Props:
 * - msg { role, text }
 * - playAudio (fn)
 * - isPlaying (bool)
 */
export default function MessageBubble({ msg, playAudio, isPlaying }) {
  const isUser = msg.role === "user";

  return (
    <div className={`my-3 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`p-3 rounded-2xl text-sm md:text-base shadow-md max-w-[75%] ${
          isUser
            ? "bg-blue-500 text-white rounded-br-none"
            : "bg-green-100 text-gray-900 rounded-bl-none dark:bg-gray-800 dark:text-gray-100"
        }`}
      >
        <div className="flex items-start gap-2">
          <div className="flex-1 whitespace-pre-wrap">{msg.text}</div>

          {!isUser && (
            <button
              onClick={playAudio}
              className="ml-2 text-blue-600 hover:text-blue-800"
              title="Play voice"
            >
              {isPlaying ? "⏸" : "🔊"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
