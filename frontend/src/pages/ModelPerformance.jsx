import { useEffect, useState } from 'react';
import { getModelInfo } from '../services/api';

const visualizations = [
  { file: 'accuracy_curve.png', label: 'Training accuracy' },
  { file: 'loss_curve.png', label: 'Training loss' },
  { file: 'confusion_matrix.png', label: 'Confusion matrix' },
  { file: 'class_distribution.png', label: 'Class distribution' },
];

export default function ModelPerformance() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo().then((response) => setInfo(response.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;
  if (!info) return <div className="error-message"><strong>Model information unavailable</strong><span>Confirm that the backend is running.</span></div>;

  const metrics = [
    { label: 'Test accuracy', value: info.test_accuracy },
    { label: 'Precision', value: info.precision_weighted },
    { label: 'Recall', value: info.recall_weighted },
    { label: 'F1 score', value: info.f1_weighted },
  ];

  return (
    <div>
      <header className="page-header">
        <div className="eyebrow"><span /> Transparent evaluation</div>
        <h1>Model performance</h1>
        <p>Actual evaluation results from the held-out test set—not estimated or fabricated metrics.</p>
      </header>

      <section className="grid grid-2">
        <article className="card">
          <span className="section-kicker">Architecture</span>
          <h2 className="panel-title" style={{ marginTop: 7 }}>Model details</h2>
          <table>
            <tbody>
              <tr><th>Base model</th><td><strong>{info.model_name}</strong></td></tr>
              <tr><th>Method</th><td>ImageNet transfer learning</td></tr>
              <tr><th>Classes</th><td>{info.number_of_classes}</td></tr>
              <tr><th>Input shape</th><td>{info.image_size} × {info.image_size} pixels</td></tr>
            </tbody>
          </table>
        </article>

        <article className="card">
          <span className="section-kicker">Held-out test set</span>
          <h2 className="panel-title" style={{ marginTop: 7 }}>Evaluation metrics</h2>
          <div className="grid grid-2">
            {metrics.map((metric) => (
              <div key={metric.label} className="metric-tile">
                <strong>{metric.value != null ? `${(metric.value * 100).toFixed(1)}%` : '—'}</strong>
                <span>{metric.label}</span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="card" style={{ marginTop: 20 }}>
        <span className="section-kicker">Evidence</span>
        <h2 className="panel-title" style={{ marginTop: 7 }}>Training and evaluation visuals</h2>
        <div className="visualization-grid">
          {visualizations.map((item) => (
            <figure className="visualization-card" key={item.file}>
              <p>{item.label}</p>
             <img
  src={`https://plant-disease-detection-cwoo.onrender.com/api/results/${item.file}`}
  alt={item.label}
  onError={(event) => {
    event.currentTarget.closest('figure').style.display = 'none';
  }}
/>
            </figure>
          ))}
        </div>
      </section>

      <section className="card" style={{ marginTop: 20 }}>
        <span className="section-kicker">Coverage</span>
        <h2 className="panel-title" style={{ marginTop: 7 }}>Recognized classes</h2>
        <div className="class-list">
          {info.class_names?.map((name) => {
            const healthy = name.toLowerCase().includes('healthy');
            return <span key={name} className={`badge ${healthy ? 'badge-healthy' : 'badge-diseased'}`}><i /> {name.replace(/___/g, ' · ').replace(/__/g, ' · ').replace(/_/g, ' ')}</span>;
          })}
        </div>
      </section>
    </div>
  );
}
