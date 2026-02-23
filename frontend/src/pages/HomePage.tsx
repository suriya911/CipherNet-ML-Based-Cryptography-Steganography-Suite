export function HomePage() {
  return (
    <section className="panel">
      <h2>Pipeline Pages</h2>
      <p className="lead">
        This frontend ships the full operational flow: secure embed, decode from model recovery, and
        stego detection confidence scoring.
      </p>
      <div className="info-grid">
        <article>
          <h3>Embed</h3>
          <p>Upload cover image + secret file, then receive generated stego image and crypto metadata.</p>
        </article>
        <article>
          <h3>Decode</h3>
          <p>Upload stego image and metadata to recover and download the decrypted secret payload.</p>
        </article>
        <article>
          <h3>Detect</h3>
          <p>Upload a candidate image and inspect cover/stego probabilities from detector inference.</p>
        </article>
      </div>
      <code className="mono-line">
        API base URL: <strong>{import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"}</strong>
      </code>
    </section>
  );
}
