import { FormEvent, useState } from "react";
import { decodeSecret } from "../apiClient";

function b64ToBlob(base64: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: "application/octet-stream" });
}

export function DecodePage() {
  const [stegoImage, setStegoImage] = useState<File | null>(null);
  const [keyHex, setKeyHex] = useState("");
  const [nonceHex, setNonceHex] = useState("");
  const [tagHex, setTagHex] = useState("");
  const [ciphertextLength, setCiphertextLength] = useState("");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [secretLength, setSecretLength] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!stegoImage) {
      setError("Choose stego image.");
      return;
    }
    const parsedLength = Number(ciphertextLength);
    if (!Number.isFinite(parsedLength) || parsedLength < 0) {
      setError("ciphertext_length must be a non-negative number.");
      return;
    }
    setBusy(true);
    setError(null);
    setDownloadUrl(null);
    setSecretLength(null);

    try {
      const data = await decodeSecret({
        stegoImage,
        keyHex: keyHex.trim(),
        nonceHex: nonceHex.trim(),
        tagHex: tagHex.trim(),
        ciphertextLength: parsedLength,
      });
      const blob = b64ToBlob(data.secret_base64);
      setDownloadUrl(URL.createObjectURL(blob));
      setSecretLength(data.secret_length);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Decode failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <h2>Decode Secret</h2>
      <form className="stack" onSubmit={onSubmit}>
        <label>
          Stego image
          <input type="file" accept="image/*" onChange={(e) => setStegoImage(e.target.files?.[0] ?? null)} />
        </label>
        <label>
          key_hex
          <input value={keyHex} onChange={(e) => setKeyHex(e.target.value)} required />
        </label>
        <label>
          nonce_hex
          <input value={nonceHex} onChange={(e) => setNonceHex(e.target.value)} required />
        </label>
        <label>
          tag_hex
          <input value={tagHex} onChange={(e) => setTagHex(e.target.value)} required />
        </label>
        <label>
          ciphertext_length
          <input value={ciphertextLength} onChange={(e) => setCiphertextLength(e.target.value)} required />
        </label>
        <button disabled={busy} type="submit">
          {busy ? "Decoding..." : "Run Decode"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {downloadUrl && (
        <div className="kv">
          <p>
            <strong>Recovered bytes:</strong> {secretLength}
          </p>
          <a className="secondary-button" href={downloadUrl} download="decoded_secret.bin">
            Download Secret
          </a>
        </div>
      )}
    </section>
  );
}
