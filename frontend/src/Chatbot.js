import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

function Chatbot({ onLogout }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const chatEndRef = useRef(null);

    // Scroll to bottom on new message
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    // Add initial welcome message once
    useEffect(() => {
        const welcomeMsg = {
            sender: "bot",
            text: "How may I assist you today?",
            timestamp: new Date(),
            welcome: true,
        };
        setMessages([welcomeMsg]);
    }, []);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMsg = { sender: "user", text: input, timestamp: new Date() };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        try {
            const res = await fetch("http://backend:5000/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: input }),
            });
            const data = await res.json();
            const botMsg = { sender: "bot", text: data.response || data, timestamp: new Date() };
            setMessages((prev) => [...prev, botMsg]);
        } catch (err) {
            setMessages((prev) => [
                ...prev,
                { sender: "bot", text: "⚠️ Could not connect to server.", timestamp: new Date() },
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app-container">
            <header className="header">
                <span className="chat-icon">💬</span> Teraleads Chatbot
                <button className="logout-btn" onClick={onLogout}>
                    Logout
                </button>
            </header>

            <div className="chat-box">
                {messages.map((msg, i) => (
                    <div key={i} className={`message ${msg.sender}`}>
                        {/* Inline icon */}
                        <span className="message-icon">
                            {msg.sender === "bot" ? "🤖" : "🧑‍💼"}
                        </span>

                        <div className="message-text">
                            <ReactMarkdown>{msg.text ? msg.text.replace(/\{[^}]*\}/g, "") : ""}</ReactMarkdown>
                        </div>

                        {/* Timestamp */}
                        <span className="timestamp">
                            {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                    </div>
                ))}
                {loading && <div className="message bot typing">Bot is typing...</div>}
                <div ref={chatEndRef} />
            </div>

            <div className="input-box">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                    placeholder="Send a request..."
                />
                <button onClick={sendMessage}>Send</button>
            </div>
        </div>
    );
}

export default Chatbot;
