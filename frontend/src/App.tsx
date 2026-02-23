import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { EmbedPage } from "./pages/EmbedPage";
import { DecodePage } from "./pages/DecodePage";
import { DetectPage } from "./pages/DetectPage";
import { HomePage } from "./pages/HomePage";

export function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="brand-kicker">CipherNet Suite</p>
          <h1>Steganography Control Center</h1>
        </div>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/embed">Embed</NavLink>
          <NavLink to="/decode">Decode</NavLink>
          <NavLink to="/detect">Detect</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/embed" element={<EmbedPage />} />
          <Route path="/decode" element={<DecodePage />} />
          <Route path="/detect" element={<DetectPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
