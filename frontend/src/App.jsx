import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Predict from './pages/Predict';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import ModelPerformance from './pages/ModelPerformance';
import DiseaseInfo from './pages/DiseaseInfo';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar />
        <main className="page-shell container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/model" element={<ModelPerformance />} />
            <Route path="/diseases" element={<DiseaseInfo />} />
          </Routes>
        </main>
        <footer className="site-footer">
          <div className="container footer-inner">
            <div>
              <strong>LeafLens AI</strong>
              <p>Plant disease detection powered by MobileNetV2.</p>
            </div>
            <span>AI-assisted academic project</span>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}
