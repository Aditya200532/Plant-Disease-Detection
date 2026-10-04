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
      <Navbar />
      <main className="container" style={{ paddingTop: 24, paddingBottom: 40 }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
          <Route path="/model" element={<ModelPerformance />} />
          <Route path="/diseases" element={<DiseaseInfo />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
