import { useState, useRef } from 'react';
import { predictImage } from '../services/api';

export default function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(file);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Predict Disease</h1>
        <p>Upload a plant leaf image to get a disease prediction</p>
      </div>

      <div className="grid grid-2">
        <div className="card">
          {!preview ? (
            <div
              className="drop-zone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => inputRef.current.click()}
            >
              <p style={{ fontSize: 40 }}>&#128196;</p>
              <p><strong>Drag &amp; drop</strong> an image here</p>
              <p>or click to browse</p>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div style={{ textAlign: 'center' }}>
              <img src={preview} alt="Preview" className="image-preview" />
              <p style={{ marginTop: 8, fontSize: 13, color: 'var(--text-secondary)' }}>{file.name}</p>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16 }}>
                <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                  {loading ? 'Analyzing...' : 'Predict'}
                </button>
                <button className="btn btn-danger" onClick={reset}>Remove</button>
              </div>
            </div>
          )}

          {loading && (
            <div className="loading" style={{ marginTop: 16 }}>
              <div className="spinner" />
            </div>
          )}
          {error && <p style={{ color: 'var(--danger)', marginTop: 12 }}>{error}</p>}
        </div>

        {result && (
          <div className="card">
            <h2 style={{ marginBottom: 16 }}>Prediction Result</h2>
            <table>
              <tbody>
                <tr><th>Plant</th><td>{result.plant}</td></tr>
                <tr><th>Disease</th><td>{result.disease}</td></tr>
                <tr>
                  <th>Status</th>
                  <td>
                    <span className={`badge ${result.status === 'Healthy' ? 'badge-healthy' : 'badge-diseased'}`}>
                      {result.status}
                    </span>
                  </td>
                </tr>
                <tr><th>Confidence</th><td>{(result.confidence * 100).toFixed(1)}%</td></tr>
              </tbody>
            </table>

            {result.warning && (
              <div className="disclaimer" style={{ marginTop: 16 }}>{result.warning}</div>
            )}

            {result.gradcam && (
              <div style={{ marginTop: 20 }}>
                <h3 style={{ marginBottom: 8, fontSize: 16 }}>Grad-CAM Visualization</h3>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
                  Highlighted regions influenced the model's prediction
                </p>
                <img
                  src={`data:image/png;base64,${result.gradcam}`}
                  alt="Grad-CAM"
                  style={{ maxWidth: '100%', borderRadius: 8 }}
                />
              </div>
            )}

            <div className="disclaimer">
              This is an AI-assisted result and should not replace professional agricultural diagnosis.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
