import React, { useState } from "react";
import { FiSun, FiMoon, FiPlus, FiTrash2, FiMenu } from "react-icons/fi";

export default function Sidebar({
  chats,
  activeChatId,
  setActiveChatId,
  createNewChat,
  deleteChat,
  currentPDF,
  uploadPDF,
  darkMode,
  setDarkMode,
}) {
  const [collapsed, setCollapsed] = useState(false);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (file) uploadPDF(file);
    e.target.value = "";
  };

  return (
    <aside
      className={`flex flex-col min-h-screen overflow-hidden transition-all duration-200 bg-white dark:bg-gray-800 border-r dark:border-gray-700 ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      <div className="flex items-center justify-between p-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <FiMenu className="text-lg text-gray-700 dark:text-gray-200" />
          </button>

          {!collapsed && (
            <div className="text-lg font-semibold text-gray-900 dark:text-white">
              Kannada RAG
            </div>
          )}
        </div>

        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          {darkMode ? (
            <FiMoon className="text-gray-200" />
          ) : (
            <FiSun className="text-gray-700" />
          )}
        </button>
      </div>

      <div className="p-2">
        <button
          onClick={createNewChat}
          className="flex items-center gap-2 w-full rounded px-3 py-2 bg-green-600 hover:bg-green-700 text-white"
        >
          <FiPlus />
          {!collapsed && <span>ಹೊಸ ಸಂಭಾಷಣೆ</span>}
        </button>
      </div>

      <div className="px-3">
        <label>
          <div className="w-full cursor-pointer bg-blue-600 text-white py-2 px-3 rounded text-center">
            {!collapsed ? "📄 PDF ಅಪ್‌ಲೋಡ್ ಮಾಡಿ" : "📄"}
          </div>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFile}
            className="hidden"
          />
        </label>

        {!collapsed && (
          <div className="mt-2 text-sm text-gray-700 dark:text-gray-300">
            ಆಯ್ಕೆ ಮಾಡಿದ PDF:
            <div className="font-medium text-gray-900 dark:text-white">
              {currentPDF || "None"}
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto mt-3 p-2">
        {Object.entries(chats).map(([id, chat]) => {
          const active = id === activeChatId;

          return (
            <div
              key={id}
              onClick={() => {
                setActiveChatId(id);
                chat.pdf = currentPDF; // 🔥 IMPORTANT FIX
              }}
              className={`flex items-center gap-2 p-2 mb-2 rounded cursor-pointer ${
                active
                  ? "bg-gray-100 dark:bg-gray-700"
                  : "bg-transparent hover:bg-gray-50 dark:hover:bg-gray-800"
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                  {chat.title || "New Chat"}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-300 truncate">
                  {chat.pdf ? `PDF: ${chat.pdf}` : "No PDF"}
                </div>
              </div>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(id);
                }}
                className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <FiTrash2 className="text-red-600 dark:text-red-400" />
              </button>
            </div>
          );
        })}

        <div className="p-3 text-xs text-gray-500 dark:text-gray-300">
          {!collapsed && (
            <div>
              Tip: Upload a PDF, ask questions, and press 🔊 to hear answers.
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
