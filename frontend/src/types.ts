export interface EmbedResponse {
  stego_image_base64: string;
  nonce_hex: string;
  tag_hex: string;
  key_hex: string;
  ciphertext_length: number;
  secret_capacity_bytes: number;
  original_secret_length: number;
}

export interface DecodeResponse {
  secret_base64: string;
  secret_length: number;
}

export interface DetectResponse {
  cover_probability: number;
  stego_probability: number;
  predicted_label: "cover" | "stego";
}
