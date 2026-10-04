import { useState, useEffect } from 'react';
import { getDiseaseInfo } from '../services/api';

export default function DiseaseInfo() {
  const [diseases, setDiseases] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDiseaseInfo().then(r => setDiseases(r.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;
  if (!diseases) return <p>Could not load disease information.</p>;

  return (
    <div>
      <div className="page-header">
        <h1>Disease Information</h1>
        <p>General educational information about plant diseases detected by the model</p>
      </div>

      <div className="grid grid-2">
        {Object.entries(diseases).map(([name, info]) => (
          <div key={name} className="card">
            <h3 style={{ marginBottom: 8 }}>
              <span className={`badge ${name === 'Healthy' ? 'badge-healthy' : 'badge-diseased'}`} style={{ marginRight: 8 }}>
                {name === 'Healthy' ? 'Healthy' : 'Disease'}
              </span>
              {name}
            </h3>
            <p style={{ fontSize: 14, marginBottom: 12 }}>{info.description}</p>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              <strong>Prevention:</strong> {info.prevention}
            </p>
          </div>
        ))}
      </div>

      <div className="disclaimer" style={{ marginTop: 20 }}>
        This information is for general educational purposes only. Do not use it as a substitute for professional agricultural advice.
        Do not apply specific pesticides or chemical treatments based solely on this information.
      </div>
    </div>
  );
}
