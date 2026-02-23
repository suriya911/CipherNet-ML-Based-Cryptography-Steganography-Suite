from pydantic import BaseModel, Field


class EmbedResponse(BaseModel):
    stego_image_base64: str = Field(..., description="Base64-encoded PNG stego image.")
    nonce_hex: str = Field(..., description="AES-GCM nonce (12 bytes) encoded as hex.")
    tag_hex: str = Field(..., description="AES-GCM authentication tag encoded as hex.")
    key_hex: str = Field(..., description="AES-256 key encoded as hex.")
    ciphertext_length: int = Field(..., description="Ciphertext length in bytes embedded in secret tensor.")
    secret_capacity_bytes: int = Field(..., description="Maximum embeddable ciphertext bytes.")
    original_secret_length: int = Field(..., description="Original plaintext secret byte length.")


class DecodeResponse(BaseModel):
    secret_base64: str = Field(..., description="Recovered plaintext secret bytes encoded as base64.")
    secret_length: int = Field(..., description="Recovered plaintext secret byte length.")


class DetectResponse(BaseModel):
    cover_probability: float = Field(..., description="Probability image is a clean cover.")
    stego_probability: float = Field(..., description="Probability image contains hidden payload.")
    predicted_label: str = Field(..., description="Predicted class label: cover or stego.")
