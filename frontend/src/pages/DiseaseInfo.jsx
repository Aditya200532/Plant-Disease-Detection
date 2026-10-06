import { useEffect, useState } from 'react';
import { getDiseaseInfo } from '../services/api';

export default function DiseaseInfo() {
  const [diseases, setDiseases] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDiseaseInfo().then((response) => setDiseases(response.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;
  if (!diseases) return <div className="error-message"><strong>Disease library unavailable</strong><span>Confirm that the backend is running.</span></div>;

  return (
    <div>
      <header className="page-header page-header-split">
        <div>
          <div className="eyebrow"><span /> Educational resource</div>
          <h1>Disease library</h1>
          <p>Understand common visual symptoms and general prevention practices.</p>
        </div>
        <div className="security-note">
          <span>i</span>
          <div><strong>General guidance</strong><small>Not a chemical treatment recommendation</small></div>
        </div>
      </header>

      <section className="grid grid-2">
        {Object.entries(diseases).map(([name, info]) => {
          const healthy = name === 'Healthy';
          return (
            <article key={name} className="card disease-card">
              <h3><span className={`badge ${healthy ? 'badge-healthy' : 'badge-diseased'}`}><i /> {healthy ? 'Healthy' : 'Disease'}</span>{name}</h3>
              <p>{info.description}</p>
              <p className="prevention"><strong>General prevention:</strong> {info.prevention}</p>
            </article>
          );
        })}
      </section>

      <div className="disclaimer" style={{ marginTop: 20 }}>
        This content is for education only. Consult an agricultural professional before deciding on treatment, pesticide, or dosage.
      </div>
    </div>
  );
}
