import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'Home' },
  { to: '/predict', label: 'Predict' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/history', label: 'History' },
  { to: '/model', label: 'Model' },
  { to: '/diseases', label: 'Diseases' },
];

export default function Navbar() {
  return (
    <nav style={{
      background: '#fff',
      borderBottom: '1px solid #e0e0e0',
      padding: '0 20px',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      <div className="container" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: 60,
        flexWrap: 'wrap',
      }}>
        <NavLink to="/" style={{ fontWeight: 700, fontSize: 18, color: '#2e7d32', textDecoration: 'none' }}>
          Plant Disease Detection
        </NavLink>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {links.map(link => (
            <NavLink
              key={link.to}
              to={link.to}
              style={({ isActive }) => ({
                padding: '8px 14px',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 500,
                background: isActive ? '#e8f5e9' : 'transparent',
                color: isActive ? '#2e7d32' : '#5a5a7a',
                textDecoration: 'none',
                transition: 'background 0.2s',
              })}
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}
