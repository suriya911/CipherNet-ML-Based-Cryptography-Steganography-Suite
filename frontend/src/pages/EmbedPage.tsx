import { FormEvent, useMemo, useState } from "react";
import { embedSecret, toDataUrl } from "../apiClient";
import type { EmbedResponse } from "../types";

export function EmbedPage() {
  const [coverImage, setCoverImage] = useState<File | null>(null);
  const [secretFile, setSecretFile] = useState<File | null>(null);
  const [keyHex, setKeyHex] = useState("");
  const [result, setResult] = useState<EmbedResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stegoImageUrl = useMemo(() => {
    if (!result) return null;
    return toDataUrl(result.stego_image_base64);
  }, [result]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!coverImage || !secretFile) {
      setError("Choose both cover image and secret file.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await embedSecret({
        coverImage,
        secretFile,
        keyHex: keyHex.trim() || undefined,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Embed failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <h2>Embed Secret</h2>
      <form className="stack" onSubmit={onSubmit}>
        <label>
          Cover image
          <input type="file" accept="image/*" onChange={(e) => setCoverImage(e.target.files?.[0] ?? null)} />
        </label>
        <label>
          Secret file
          <input type="file" onChange={(e) => setSecretFile(e.target.files?.[0] ?? null)} />
        </label>
        <label>
          AES key hex (optional, 64 hex chars)
          <input
            type="text"
            placeholder="Leave empty for random key"
            value={keyHex}
            onChange={(e) => setKeyHex(e.target.value)}
          />
        </label>
        <button disabled={busy} type="submit">
          {busy ? "Embedding..." : "Run Embed"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && stegoImageUrl && (
        <div className="result-grid">
          <img src={stegoImageUrl} alt="Stego output" />
          <div className="kv">
            <p>
              <strong>key_hex:</strong> <code>{result.key_hex}</code>
            </p>
            <p>
              <strong>nonce_hex:</strong> <code>{result.nonce_hex}</code>
            </p>
            <p>
              <strong>tag_hex:</strong> <code>{result.tag_hex}</code>
            </p>
            <p>
              <strong>ciphertext_length:</strong> {result.ciphertext_length}
            </p>
            <p>
              <strong>secret_capacity_bytes:</strong> {result.secret_capacity_bytes}
            </p>
            <a className="secondary-button" href={stegoImageUrl} download="stego.png">
              Download Stego PNG
            </a>
          </div>
        </div>
      )}
    </section>
  );
}
