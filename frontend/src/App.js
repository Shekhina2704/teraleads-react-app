import React, { useState } from "react";
import Auth from "./Auth";
import Chatbot from "./Chatbot";
import "./App.css";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem("token"));

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsAuthenticated(false);
  };

  // Background style for public/bgnd.jpg
  const backgroundStyle = {
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    backgroundImage: `url(${process.env.PUBLIC_URL + "/bgnd.jpg"})`,
    backgroundSize: "cover",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
  };

  return (
    <div style={backgroundStyle}>
      {isAuthenticated ? (
        <Chatbot onLogout={handleLogout} />
      ) : (
        <Auth onAuthSuccess={() => setIsAuthenticated(true)} />
      )}
    </div>
  );
}

export default App;
