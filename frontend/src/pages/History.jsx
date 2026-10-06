import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getHistory } from '../services/api';

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory().then((response) => setHistory(response.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;

  return (
    <div>
      <header className="page-header page-header-split">
        <div>
          <div className="eyebrow"><span /> Saved activity</div>
          <h1>Prediction history</h1>
          <p>Review recent plant health analyses stored in MongoDB.</p>
        </div>
        <Link to="/predict" className="btn btn-primary">New diagnosis <span>→</span></Link>
      </header>

      <section className="card table-card">
        {history.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">AI</div>
            <h3>Your history is empty</h3>
            <p>Complete your first leaf diagnosis to start building a searchable prediction record.</p>
            <Link to="/predict" className="btn btn-primary">Analyze a leaf</Link>
          </div>
        ) : (
          <table>
            <thead>
              <tr><th>Date</th><th>Plant</th><th>Disease</th><th>Status</th><th>Confidence</th></tr>
            </thead>
            <tbody>
              {history.map((item, index) => (
                <tr key={`${item.timestamp}-${index}`}>
                  <td>{item.timestamp ? new Date(item.timestamp).toLocaleString() : '—'}</td>
                  <td><strong>{item.plant}</strong></td>
                  <td>{item.disease}</td>
                  <td><span className={`badge ${item.status === 'Healthy' ? 'badge-healthy' : 'badge-diseased'}`}><i /> {item.status}</span></td>
                  <td><strong>{(item.confidence * 100).toFixed(1)}%</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
