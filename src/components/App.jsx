import React, { useState, useEffect, useRef } from "react";
import { MessageSquare, Send, Menu, X, User, Bot } from "lucide-react";
import ReactMarkdown from "react-markdown";
import "../assets/styles/index.css";
 // For gradient background and font

function App() {
  const [histories, setHistories] = useState([]);
  const [input, setInput] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(localStorage.getItem("sessionId"));
  const [chatHistory, setChatHistory] = useState([]);
  const chatRef = useRef(null);

  useEffect(() => {
    const storedHistories = JSON.parse(localStorage.getItem("chatHistories")) || [];
    setHistories(storedHistories);

    if (!sessionId) {
      startNewSession();
    } else {
      fetchChatHistory(sessionId);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("chatHistories", JSON.stringify(histories));
  }, [histories]);

  useEffect(() => {
    if (sessionId) {
      fetchChatHistory(sessionId);
    }
  }, [sessionId]);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const startNewSession = async () => {
    try {
      const response = await fetch("http://localhost:5000/chat/start", { method: "POST" });
      const data = await response.json();
      setSessionId(data.session_id);
      localStorage.setItem("sessionId", data.session_id);
      const newHistories = [...histories, { sessionId: data.session_id, title: "" }];
      setHistories(newHistories);
      setChatHistory([]);
    } catch (error) {
      console.error("Error starting new session:", error);
    }
  };

  const fetchChatHistory = async (id) => {
    try {
      const response = await fetch(`http://localhost:5000/chat/history/${id}`);
      const data = await response.json();
      setChatHistory(data.chat_history || []);
    } catch (error) {
      console.error("Error fetching chat history:", error);
    }
  };

  const handleSessionChange = (id) => {
    setSessionId(id);
    localStorage.setItem("sessionId", id);
    fetchChatHistory(id);
  };

  const handleSend = async () => {
    if (!input.trim() || !sessionId) return;

    const newMessage = { query: input, response: "..." };
    setChatHistory((prev) => [...prev, newMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input, session_id: sessionId }),
      });

      const data = await response.json();
      const botMessage = { query: input, response: data.response };
      setChatHistory((prev) => [...prev.slice(0, prev.length - 1), botMessage]);

      setHistories((prev) =>
        prev.map((h) =>
          h.sessionId === sessionId && !h.title ? { ...h, title: input } : h
        )
      );
    } catch (error) {
      console.error("Error fetching response:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen font-poppins bg-gradient-to-br from-blue-50 via-white to-orange-50">
      {/* Sidebar */}
      <div
        className={`${
          isSidebarOpen ? "w-64" : "w-0"
        } bg-white border-r border-gray-200 shadow-lg transition-all duration-300 overflow-hidden flex flex-col`}
      >
        <div className="p-4 border-b border-gray-200">
          <button
            onClick={startNewSession}
            className="w-full bg-blue-500 text-white rounded-lg py-2 flex items-center justify-center gap-2 hover:bg-blue-600 transition"
          >
            <MessageSquare size={18} /> New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {histories.map((history, index) => (
            <button
              key={index}
              onClick={() => handleSessionChange(history.sessionId)}
              className={`w-full text-left p-3 hover:bg-blue-50 ${
                sessionId === history.sessionId ? "bg-blue-100" : ""
              }`}
            >
              <div className="font-medium text-gray-700 truncate">
                {history.title || `Chat ${index + 1}`}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="bg-white border-b border-gray-200 p-4 flex items-center shadow-sm">
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 hover:bg-gray-100 rounded-lg text-gray-700"
          >
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <h1 className="ml-4 text-xl font-bold text-blue-700">
            IIIT Dharwad Copilot
          </h1>
        </div>

        {/* Messages */}
        <div ref={chatRef} className="flex-1 overflow-y-auto p-6 space-y-4">
          {chatHistory.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-blue-600 mb-2">
                  🎓 Welcome to IIIT Dharwad Copilot!
                </h2>
                <p className="text-gray-600">
                  Ask me anything about admissions, courses, faculty, or campus life.
                </p>
              </div>
            </div>
          ) : (
            chatHistory.map((message, index) => (
              <React.Fragment key={index}>
                {/* User Message */}
                <div className="flex justify-end">
                  <div className="max-w-lg bg-blue-500 text-white p-3 rounded-2xl shadow">
                    {message.query}
                  </div>
                </div>

                {/* Bot Message */}
                <div className="flex justify-start">
                  <div className="max-w-lg bg-yellow-100 text-gray-800 p-3 rounded-2xl shadow">
                    <ReactMarkdown
                      components={{
                        ul: ({ children }) => <ul className="list-disc list-inside space-y-1">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal list-inside space-y-1">{children}</ol>,
                        li: ({ children }) => <li>{children}</li>,
                        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                        p: ({ children }) => <p className="mb-2">{children}</p>,
                      }}
                    >
                      {message.response}
                    </ReactMarkdown>
                  </div>
                </div>
              </React.Fragment>
            ))
          )}
        </div>

        {/* Input Box */}
        <div className="bg-white border-t border-gray-200 p-4 shadow-lg">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Type your question about IIIT Dharwad..."
              className="flex-1 border border-gray-300 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
            />
            <button
              onClick={handleSend}
              className="bg-orange-500 text-white px-4 py-2 rounded-lg hover:bg-orange-600 transition"
            >
              {loading ? "Loading..." : <Send size={18} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
