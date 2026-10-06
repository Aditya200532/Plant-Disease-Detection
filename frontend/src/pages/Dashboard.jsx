import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getStats } from '../services/api';

const COLORS = ['#65a85c', '#b84b45'];

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats().then((response) => setStats(response.data)).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading"><div className="spinner" /></div>;
  if (!stats) return <div className="error-message"><strong>Dashboard unavailable</strong><span>Confirm that the backend is running.</span></div>;

  const pieData = [
    { name: 'Healthy', value: stats.healthy },
    { name: 'Diseased', value: stats.diseased },
  ].filter((item) => item.value > 0);

  const summary = [
    { value: stats.total, label: 'Total scans' },
    { value: stats.healthy, label: 'Healthy results' },
    { value: stats.diseased, label: 'Disease results' },
    { value: stats.top_diseases?.[0]?.disease || '—', label: 'Most detected' },
  ];

  return (
    <div>
      <header className="page-header page-header-split">
        <div>
          <div className="eyebrow"><span /> Live overview</div>
          <h1>Health dashboard</h1>
          <p>Track prediction activity and the conditions identified most often.</p>
        </div>
        <span className={`badge ${stats.db_available ? 'badge-healthy' : 'badge-diseased'}`}>
          <i /> {stats.db_available ? 'Database connected' : 'Database offline'}
        </span>
      </header>

      {!stats.db_available && (
        <div className="disclaimer" style={{ marginBottom: 20 }}>
          MongoDB is not connected. Statistics will populate when the database is available and new predictions are made.
        </div>
      )}

      <section className="grid grid-4">
        {summary.map((item) => (
          <article className="card stat-card" key={item.label}>
            <h3>{item.value}</h3>
            <p>{item.label}</p>
          </article>
        ))}
      </section>

      {stats.total === 0 ? (
        <section className="card empty-state" style={{ marginTop: 20 }}>
          <div className="empty-state-icon">00</div>
          <h3>No prediction data yet</h3>
          <p>Once MongoDB is connected, every completed leaf analysis will contribute to these charts.</p>
        </section>
      ) : (
        <section className="grid grid-2" style={{ marginTop: 20 }}>
          <article className="card chart-card">
            <h2 className="panel-title">Plant health split</h2>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={62} outerRadius={98} paddingAngle={4} dataKey="value">
                  {pieData.map((_, index) => <Cell key={index} fill={COLORS[index % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </article>

          <article className="card chart-card">
            <h2 className="panel-title">Top detected diseases</h2>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={stats.top_diseases} margin={{ left: 0, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8df" />
                <XAxis dataKey="disease" tick={{ fontSize: 10, fill: '#66746d' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: '#66746d' }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: '#f1f5ec' }} />
                <Bar dataKey="count" fill="#216548" radius={[7, 7, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </article>
        </section>
      )}
    </div>
  );
}
