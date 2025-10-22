import React, { useState } from "react";

function Auth({ onAuthSuccess }) {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ full_name: "", email: "", password: "" });
    const [message, setMessage] = useState("");

    const toggleMode = () => {
        setIsLogin(!isLogin);
        setMessage("");
        setFormData({ full_name: "", email: "", password: "" });
    };

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const endpoint = isLogin ? "http://localhost:5000/api/login" : "http://localhost:5000/api/signup";

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            const data = await response.json();

            if (response.ok) {
                localStorage.setItem("token", data.token || "");
                setMessage(isLogin ? "✅ Login successful!" : "✅ Account created!");
                onAuthSuccess(); // notify App that login/signup succeeded
            } else {
                setMessage(data.error || "❌ Something went wrong!");
            }
        } catch (err) {
            console.error(err);
            setMessage("⚠️ Server error, please try again.");
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-card">
                <h2>{isLogin ? "Welcome Back" : "Create Account"}</h2>
                <p className="subtitle">
                    {isLogin
                        ? "Log in to manage your dental appointments"
                        : "Sign up to start booking appointments"}
                </p>

                <form onSubmit={handleSubmit}>
                    {!isLogin && (
                        <div className="form-group">
                            <label>Full Name</label>
                            <input
                                type="text"
                                name="full_name"
                                placeholder="Enter your name"
                                value={formData.full_name}
                                onChange={handleChange}
                                required
                            />
                        </div>
                    )}

                    <div className="form-group">
                        <label>Email</label>
                        <input
                            type="email"
                            name="email"
                            placeholder="Enter your email"
                            value={formData.email}
                            onChange={handleChange}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <input
                            type="password"
                            name="password"
                            placeholder="Enter your password"
                            value={formData.password}
                            onChange={handleChange}
                            required
                        />
                    </div>

                    <button type="submit" className="btn-primary">
                        {isLogin ? "Login" : "Sign Up"}
                    </button>
                </form>

                {message && <p className="message">{message}</p>}

                <p className="toggle-text">
                    {isLogin ? "Don’t have an account?" : "Already have an account?"}{" "}
                    <span onClick={toggleMode} className="toggle-link">
                        {isLogin ? "Sign up here" : "Login"}
                    </span>
                </p>
            </div>
        </div>
    );
}

export default Auth;
