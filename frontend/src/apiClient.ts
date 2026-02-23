import type { DecodeResponse, DetectResponse, EmbedResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export async function embedSecret(params: {
  coverImage: File;
  secretFile: File;
  keyHex?: string;
}): Promise<EmbedResponse> {
  const form = new FormData();
  form.append("cover_image", params.coverImage);
  form.append("secret_file", params.secretFile);
  if (params.keyHex) {
    form.append("key_hex", params.keyHex);
  }
  const response = await fetch(`${API_BASE}/embed`, { method: "POST", body: form });
  return parseJsonOrThrow<EmbedResponse>(response);
}

export async function decodeSecret(params: {
  stegoImage: File;
  keyHex: string;
  nonceHex: string;
  tagHex: string;
  ciphertextLength: number;
}): Promise<DecodeResponse> {
  const form = new FormData();
  form.append("stego_image", params.stegoImage);
  form.append("key_hex", params.keyHex);
  form.append("nonce_hex", params.nonceHex);
  form.append("tag_hex", params.tagHex);
  form.append("ciphertext_length", String(params.ciphertextLength));
  const response = await fetch(`${API_BASE}/decode`, { method: "POST", body: form });
  return parseJsonOrThrow<DecodeResponse>(response);
}

export async function detectStego(image: File): Promise<DetectResponse> {
  const form = new FormData();
  form.append("image", image);
  const response = await fetch(`${API_BASE}/detect`, { method: "POST", body: form });
  return parseJsonOrThrow<DetectResponse>(response);
}

export function toDataUrl(base64Png: string): string {
  return `data:image/png;base64,${base64Png}`;
}
