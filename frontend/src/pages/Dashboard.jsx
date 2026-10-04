import { useState, useEffect } from 'react';
import { getStats } from '../services/api';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const COLORS = ['#4caf50', '#f44336'];

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats().then(r => setStats(r.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;
  if (!stats) return <p>Could not load statistics.</p>;

  const pieData = [
    { name: 'Healthy', value: stats.healthy },
    { name: 'Diseased', value: stats.diseased },
  ].filter(d => d.value > 0);

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Prediction statistics overview</p>
      </div>

      {!stats.db_available && (
        <div className="disclaimer" style={{ marginBottom: 20 }}>
          MongoDB is not connected. Statistics will appear once the database is available and predictions are made.
        </div>
      )}

      <div className="grid grid-4">
        <div className="card stat-card">
          <h3>{stats.total}</h3>
          <p>Total Predictions</p>
        </div>
        <div className="card stat-card">
          <h3 style={{ color: '#4caf50' }}>{stats.healthy}</h3>
          <p>Healthy</p>
        </div>
        <div className="card stat-card">
          <h3 style={{ color: '#f44336' }}>{stats.diseased}</h3>
          <p>Diseased</p>
        </div>
        <div className="card stat-card">
          <h3>{stats.top_diseases?.[0]?.disease || '-'}</h3>
          <p>Most Detected</p>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 20 }}>
        {pieData.length > 0 && (
          <div className="card">
            <h3 style={{ marginBottom: 16 }}>Healthy vs Diseased</h3>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {stats.top_diseases?.length > 0 && (
          <div className="card">
            <h3 style={{ marginBottom: 16 }}>Top Detected Diseases</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={stats.top_diseases}>
                <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f44336" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
