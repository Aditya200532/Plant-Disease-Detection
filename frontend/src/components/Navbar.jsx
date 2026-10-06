import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'Home' },
  { to: '/predict', label: 'Diagnose' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/history', label: 'History' },
  { to: '/model', label: 'Model' },
  { to: '/diseases', label: 'Library' },
];

function LeafMark() {
  return (
    <svg viewBox="0 0 44 44" aria-hidden="true">
      <path d="M35.8 7.7C23.6 8.1 13.5 12.3 9.4 21.1c-3.2 6.9.8 14.3 7.6 15.2 8.4 1.1 15.1-6.1 16.3-14.3.7-4.9 1.3-9.5 2.5-14.3Z" fill="currentColor" />
      <path d="M11.5 35.5c5.2-8.8 11.3-14.9 19.3-20" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M19.4 26.2c.1-3.3-.5-5.7-1.5-7.4M21.8 23.5c3.5.1 6-.6 7.9-1.7" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export default function Navbar() {
  return (
    <header className="site-header">
      <nav className="container navbar" aria-label="Main navigation">
        <NavLink to="/" className="brand" aria-label="LeafLens AI home">
          <span className="brand-mark"><LeafMark /></span>
          <span className="brand-copy">
            <strong>LeafLens</strong>
            <small>Plant Health AI</small>
          </span>
        </NavLink>

        <div className="nav-links">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {link.label}
            </NavLink>
          ))}
        </div>

        <NavLink to="/predict" className="nav-cta">
          Scan a leaf
          <span aria-hidden="true">→</span>
        </NavLink>
      </nav>
    </header>
  );
}
