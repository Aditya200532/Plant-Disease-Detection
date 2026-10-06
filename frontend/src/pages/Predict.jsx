import { useEffect, useRef, useState } from 'react';
import { predictImage } from '../services/api';

function UploadIcon() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path d="M24 33V12m0 0-8 8m8-8 8 8M11 30v5a5 5 0 0 0 5 5h16a5 5 0 0 0 5-5v-5" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const handleFile = (nextFile) => {
    if (!nextFile) return;
    if (!nextFile.type.startsWith('image/')) {
      setError('Please choose a valid image file.');
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError('Please choose an image smaller than 10 MB.');
      return;
    }
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const response = await predictImage(file);
      setResult(response.data);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Prediction failed. Confirm that the backend is running and try again.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const confidence = result ? Math.round(result.confidence * 1000) / 10 : 0;

  return (
    <div className="predict-page">
      <header className="page-header page-header-split">
        <div>
          <div className="eyebrow"><span /> AI leaf analysis</div>
          <h1>Check your plant's health</h1>
          <p>Upload one clear leaf image. Analysis typically takes only a few seconds.</p>
        </div>
        <div className="security-note">
          <span>✓</span>
          <div><strong>Private analysis</strong><small>Images are processed for prediction only</small></div>
        </div>
      </header>

      <div className={`diagnosis-layout${result ? ' has-result' : ''}`}>
        <section className="upload-panel card">
          <div className="panel-heading">
            <div><span className="panel-step">1</span><h2>Select leaf image</h2></div>
            <small>JPG, PNG, WEBP · Max 10 MB</small>
          </div>

          {!preview ? (
            <div
              className={`drop-zone${dragging ? ' active' : ''}`}
              onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex="0"
              onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}
            >
              <span className="upload-icon"><UploadIcon /></span>
              <h3>Drop your leaf image here</h3>
              <p>Use a well-lit photo with one leaf clearly visible</p>
              <span className="browse-button">Browse image</span>
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp"
                hidden
                onChange={(event) => handleFile(event.target.files[0])}
              />
            </div>
          ) : (
            <div className="preview-stage">
              <div className="preview-image-wrap">
                <img src={preview} alt="Selected plant leaf" className="image-preview" />
                {loading && <div className="scan-line" />}
                <span className="preview-label">Leaf preview</span>
              </div>
              <div className="file-details">
                <div>
                  <strong>{file.name}</strong>
                  <span>{(file.size / 1024 / 1024).toFixed(2)} MB · Ready for analysis</span>
                </div>
                <button className="text-button" onClick={reset} disabled={loading}>Replace</button>
              </div>
              <button className="btn btn-primary analyze-button" onClick={handlePredict} disabled={loading}>
                {loading ? <><span className="button-spinner" /> Analyzing leaf...</> : <>Run AI diagnosis <span>→</span></>}
              </button>
            </div>
          )}

          {error && <div className="error-message"><strong>Unable to analyze</strong><span>{error}</span></div>}

          <div className="photo-tips">
            <strong>For the best result</strong>
            <div><span>Natural light</span><span>Leaf in focus</span><span>Plain background</span></div>
          </div>
        </section>

        <section className={`result-panel card${result ? ' visible' : ''}`}>
          {!result ? (
            <div className="result-placeholder">
              <div className="placeholder-rings"><span>AI</span></div>
              <h2>Your diagnosis will appear here</h2>
              <p>Choose an image and run the analysis to see plant, disease, confidence, and explainability results.</p>
              <div className="placeholder-lines"><i /><i /><i /></div>
            </div>
          ) : (
            <div className="result-content">
              <div className="result-heading">
                <div>
                  <span className="section-kicker">Analysis complete</span>
                  <h2>Prediction result</h2>
                </div>
                <span className={`badge ${result.status === 'Healthy' ? 'badge-healthy' : 'badge-diseased'}`}>
                  <i /> {result.status}
                </span>
              </div>

              <div className="diagnosis-name">
                <span>{result.plant}</span>
                <strong>{result.disease}</strong>
              </div>

              <div className="confidence-card">
                <div><span>Model confidence</span><strong>{confidence}%</strong></div>
                <div className="confidence-track"><span style={{ width: `${confidence}%` }} /></div>
                <small>Confidence measures model certainty, not diagnostic certainty.</small>
              </div>

              {result.warning && <div className="warning-message">{result.warning}</div>}

              {result.gradcam && (
                <div className="gradcam-card">
                  <div><strong>Visual explanation</strong><span>Highlighted areas influenced the prediction</span></div>
                  <img src={`data:image/png;base64,${result.gradcam}`} alt="Grad-CAM model attention visualization" />
                </div>
              )}

              <div className="result-disclaimer">
                <span>i</span>
                <p>This AI-assisted result is educational and should not replace professional agricultural diagnosis.</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
