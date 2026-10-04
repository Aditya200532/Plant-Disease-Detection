import { useState, useEffect } from 'react';
import { getHistory } from '../services/api';

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory().then(r => setHistory(r.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;

  return (
    <div>
      <div className="page-header">
        <h1>Prediction History</h1>
        <p>Recent predictions stored in the database</p>
      </div>

      <div className="card" style={{ overflowX: 'auto' }}>
        {history.length === 0 ? (
          <p style={{ textAlign: 'center', padding: 20, color: 'var(--text-secondary)' }}>
            No predictions yet. Upload an image to get started.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Plant</th>
                <th>Disease</th>
                <th>Status</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item, i) => (
                <tr key={i}>
                  <td>{item.timestamp ? new Date(item.timestamp).toLocaleString() : '-'}</td>
                  <td>{item.plant}</td>
                  <td>{item.disease}</td>
                  <td>
                    <span className={`badge ${item.status === 'Healthy' ? 'badge-healthy' : 'badge-diseased'}`}>
                      {item.status}
                    </span>
                  </td>
                  <td>{(item.confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
