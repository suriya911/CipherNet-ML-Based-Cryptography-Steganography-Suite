import { FormEvent, useState } from "react";
import { detectStego } from "../apiClient";

export function DetectPage() {
  const [image, setImage] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    cover_probability: number;
    stego_probability: number;
    predicted_label: string;
  } | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!image) {
      setError("Choose an image for detection.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await detectStego(image);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detect failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <h2>Detect Stego Signature</h2>
      <form className="stack" onSubmit={onSubmit}>
        <label>
          Candidate image
          <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} />
        </label>
        <button disabled={busy} type="submit">
          {busy ? "Detecting..." : "Run Detect"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result-grid compact">
          <article className="stat-card">
            <h3>Prediction</h3>
            <p>{result.predicted_label}</p>
          </article>
          <article className="stat-card">
            <h3>Cover Probability</h3>
            <p>{(result.cover_probability * 100).toFixed(2)}%</p>
          </article>
          <article className="stat-card">
            <h3>Stego Probability</h3>
            <p>{(result.stego_probability * 100).toFixed(2)}%</p>
          </article>
        </div>
      )}
    </section>
  );
}
