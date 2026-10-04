import { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';

export default function ModelPerformance() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo().then(r => setInfo(r.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;
  if (!info) return <p>Could not load model information.</p>;

  const metrics = [
    { label: 'Test Accuracy', value: info.test_accuracy },
    { label: 'Precision', value: info.precision_weighted },
    { label: 'Recall', value: info.recall_weighted },
    { label: 'F1 Score', value: info.f1_weighted },
  ];

  return (
    <div>
      <div className="page-header">
        <h1>Model Performance</h1>
        <p>Details about the trained MobileNetV2 model</p>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3 style={{ marginBottom: 16 }}>Model Details</h3>
          <table>
            <tbody>
              <tr><th>Model</th><td>{info.model_name}</td></tr>
              <tr><th>Architecture</th><td>Transfer Learning (ImageNet)</td></tr>
              <tr><th>Number of Classes</th><td>{info.number_of_classes}</td></tr>
              <tr><th>Input Size</th><td>{info.image_size} x {info.image_size} px</td></tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: 16 }}>Evaluation Metrics</h3>
          <div className="grid grid-2">
            {metrics.map(m => (
              <div key={m.label} className="stat-card" style={{ background: '#f5f7fa', borderRadius: 8, padding: 16 }}>
                <h3 style={{ color: 'var(--primary)' }}>
                  {m.value != null ? `${(m.value * 100).toFixed(1)}%` : '-'}
                </h3>
                <p>{m.label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3 style={{ marginBottom: 16 }}>Training &amp; Evaluation Visualizations</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 16, fontSize: 14 }}>
          Generated during model training and evaluation. These images are served from the results directory.
        </p>
        <div className="grid grid-2">
          {['accuracy_curve.png', 'loss_curve.png', 'confusion_matrix.png', 'class_distribution.png'].map(name => (
            <div key={name} style={{ textAlign: 'center' }}>
              <p style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>{name.replace(/_/g, ' ').replace('.png', '')}</p>
              <img
                src={`/api/results/${name}`}
                alt={name}
                style={{ maxWidth: '100%', borderRadius: 8, border: '1px solid var(--border)' }}
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginTop: 20 }}>
        <h3 style={{ marginBottom: 16 }}>Class Names</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {info.class_names?.map(name => (
            <span key={name} className="badge" style={{
              background: name.toLowerCase().includes('healthy') ? '#e8f5e9' : '#fff3e0',
              color: name.toLowerCase().includes('healthy') ? '#2e7d32' : '#e65100',
            }}>
              {name.replace(/___/g, ' - ').replace(/__/g, ' - ').replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
