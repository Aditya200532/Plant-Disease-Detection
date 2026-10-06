import { Link } from 'react-router-dom';

function HeroLeaf() {
  return (
    <svg className="hero-leaf" viewBox="0 0 420 480" role="img" aria-label="Illustrated plant leaf being analyzed">
      <defs>
        <linearGradient id="leafFill" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#9ed56e" />
          <stop offset="0.55" stopColor="#4d9b5f" />
          <stop offset="1" stopColor="#1f6849" />
        </linearGradient>
        <filter id="leafShadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="20" stdDeviation="18" floodColor="#123b2c" floodOpacity=".24" />
        </filter>
      </defs>
      <path d="M353 38C219 48 94 100 67 213c-24 101 42 195 143 189 111-7 170-116 156-223-7-55-15-98-13-141Z" fill="url(#leafFill)" filter="url(#leafShadow)" />
      <path d="M92 389c57-114 132-208 235-296" fill="none" stroke="#e8f7c9" strokeWidth="8" strokeLinecap="round" opacity=".84" />
      <path d="M162 275c-4-50-17-86-42-116M194 230c50-1 89-14 120-38M230 185c-1-33-8-59-25-80M134 324c42-2 76-11 104-30" fill="none" stroke="#d9f0b7" strokeWidth="5" strokeLinecap="round" opacity=".72" />
      <g fill="none" stroke="#dff7c8" strokeWidth="3">
        <path d="M40 94V54h40M340 54h40v40M40 383v40h40M380 383v40h-40" />
      </g>
      <circle cx="269" cy="164" r="24" fill="#fff" opacity=".16" />
      <circle cx="269" cy="164" r="12" fill="none" stroke="#fff" strokeWidth="2" opacity=".85" />
      <path d="m278 173 15 15" stroke="#fff" strokeWidth="3" strokeLinecap="round" opacity=".85" />
    </svg>
  );
}

const steps = [
  { number: '01', title: 'Upload a leaf', text: 'Choose a clear photo showing the complete leaf surface.' },
  { number: '02', title: 'AI analyzes it', text: 'MobileNetV2 examines visual patterns linked to plant health.' },
  { number: '03', title: 'Review the result', text: 'See the predicted condition, confidence, and visual explanation.' },
];

export default function Home() {
  return (
    <div className="home-page">
      <section className="hero-section">
        <div className="hero-copy">
          <div className="eyebrow"><span /> Deep learning for healthier crops</div>
          <h1>Know what your plant needs, <em>one leaf at a time.</em></h1>
          <p className="hero-description">
            Upload a leaf image and receive an instant AI-assisted disease classification powered by a model trained on 20,638 PlantVillage images.
          </p>
          <div className="hero-actions">
            <Link to="/predict" className="btn btn-primary btn-large">
              Diagnose a leaf <span aria-hidden="true">→</span>
            </Link>
            <Link to="/model" className="btn btn-secondary btn-large">Explore the model</Link>
          </div>
          <div className="hero-metrics" aria-label="Project metrics">
            <div><strong>89.7%</strong><span>Test accuracy</span></div>
            <div><strong>15</strong><span>Plant classes</span></div>
            <div><strong>20,638</strong><span>Training images</span></div>
          </div>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="visual-orbit orbit-one" />
          <div className="visual-orbit orbit-two" />
          <HeroLeaf />
          <div className="floating-card floating-status">
            <span className="status-dot" />
            <div><small>AI analysis</small><strong>Ready to scan</strong></div>
          </div>
          <div className="floating-card floating-model">
            <span className="mini-icon">AI</span>
            <div><small>Powered by</small><strong>MobileNetV2</strong></div>
          </div>
        </div>
      </section>

      <section className="trust-strip">
        <span>Built with</span>
        <strong>TensorFlow</strong>
        <i />
        <strong>MobileNetV2</strong>
        <i />
        <strong>FastAPI</strong>
        <i />
        <strong>PlantVillage</strong>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <span className="section-kicker">Simple workflow</span>
            <h2>From image to insight in seconds</h2>
          </div>
          <p>No specialist equipment is required—just a clear photograph of the affected leaf.</p>
        </div>
        <div className="workflow-grid">
          {steps.map((step) => (
            <article className="workflow-card" key={step.number}>
              <span className="step-number">{step.number}</span>
              <div className="step-line" />
              <h3>{step.title}</h3>
              <p>{step.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="insight-banner">
        <div>
          <span className="section-kicker light">Responsible AI</span>
          <h2>Useful guidance, not a final diagnosis.</h2>
          <p>LeafLens combines prediction confidence with Grad-CAM visualization so you can understand which area influenced the model.</p>
        </div>
        <Link to="/diseases" className="btn btn-light">Browse disease library</Link>
      </section>
    </div>
  );
}
