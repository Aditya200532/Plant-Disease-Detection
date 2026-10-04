import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div>
      <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
        <h1 style={{ fontSize: 36, marginBottom: 12 }}>Plant Disease Detection</h1>
        <p style={{ fontSize: 18, color: 'var(--text-secondary)', maxWidth: 600, margin: '0 auto 24px' }}>
          AI-powered plant leaf disease classification using deep learning.
          Upload a clear image of a plant leaf to detect possible diseases.
        </p>
        <Link to="/predict" className="btn btn-primary" style={{ fontSize: 16, padding: '14px 32px' }}>
          Upload &amp; Predict
        </Link>
      </div>

      <div className="grid grid-3" style={{ marginTop: 24 }}>
        <div className="card stat-card">
          <h3 style={{ fontSize: 24 }}>Upload</h3>
          <p>Take or upload a clear photo of a plant leaf</p>
        </div>
        <div className="card stat-card">
          <h3 style={{ fontSize: 24 }}>Analyze</h3>
          <p>MobileNetV2 deep learning model processes the image</p>
        </div>
        <div className="card stat-card">
          <h3 style={{ fontSize: 24 }}>Results</h3>
          <p>Get disease prediction with confidence score</p>
        </div>
      </div>

      <div className="disclaimer" style={{ marginTop: 24 }}>
        This tool provides AI-assisted predictions for educational purposes. Results should not replace professional agricultural diagnosis.
      </div>
    </div>
  );
}
