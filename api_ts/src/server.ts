import cors from "cors";
import express, { Request } from "express";
import multer from "multer";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const PORT = Number(process.env.PORT ?? 9000);
const PYTHON_API_BASE_URL = process.env.PYTHON_API_BASE_URL ?? "http://127.0.0.1:8000";

app.use(cors());

function requireFile(file: Express.Multer.File | undefined, fieldName: string): Express.Multer.File {
  if (!file) {
    throw new Error(`Missing required file field: ${fieldName}`);
  }
  return file;
}

function appendFile(form: FormData, fieldName: string, file: Express.Multer.File): void {
  const arrayBuffer = new ArrayBuffer(file.buffer.length);
  new Uint8Array(arrayBuffer).set(file.buffer);
  const blob = new Blob([arrayBuffer], { type: file.mimetype || "application/octet-stream" });
  form.append(fieldName, blob, file.originalname || `${fieldName}.bin`);
}

async function forwardJson(path: string, form: FormData): Promise<Response> {
  return fetch(`${PYTHON_API_BASE_URL}${path}`, { method: "POST", body: form });
}

app.get("/health", async (_req, res) => {
  try {
    const response = await fetch(`${PYTHON_API_BASE_URL}/health`);
    const payload = (await response.json()) as Record<string, unknown>;
    res.json({ status: "ok", upstream: payload });
  } catch (error) {
    res.status(503).json({ status: "error", detail: String(error) });
  }
});

app.post(
  "/embed",
  upload.fields([
    { name: "cover_image", maxCount: 1 },
    { name: "secret_file", maxCount: 1 },
  ]),
  async (req: Request, res) => {
    try {
      const files = req.files as Record<string, Express.Multer.File[]> | undefined;
      const coverImage = requireFile(files?.cover_image?.[0], "cover_image");
      const secretFile = requireFile(files?.secret_file?.[0], "secret_file");

      const form = new FormData();
      appendFile(form, "cover_image", coverImage);
      appendFile(form, "secret_file", secretFile);
      const keyHex = req.body.key_hex as string | undefined;
      if (keyHex) {
        form.append("key_hex", keyHex);
      }

      const response = await forwardJson("/embed", form);
      const bodyText = await response.text();
      res.status(response.status).type("application/json").send(bodyText);
    } catch (error) {
      res.status(400).json({ detail: String(error) });
    }
  },
);

app.post("/decode", upload.single("stego_image"), async (req, res) => {
  try {
    const stegoImage = requireFile(req.file, "stego_image");
    const form = new FormData();
    appendFile(form, "stego_image", stegoImage);
    form.append("key_hex", req.body.key_hex as string);
    form.append("nonce_hex", req.body.nonce_hex as string);
    form.append("tag_hex", req.body.tag_hex as string);
    form.append("ciphertext_length", String(req.body.ciphertext_length));

    const response = await forwardJson("/decode", form);
    const bodyText = await response.text();
    res.status(response.status).type("application/json").send(bodyText);
  } catch (error) {
    res.status(400).json({ detail: String(error) });
  }
});

app.post("/extract", upload.single("stego_image"), async (req, res) => {
  try {
    const stegoImage = requireFile(req.file, "stego_image");
    const form = new FormData();
    appendFile(form, "stego_image", stegoImage);
    form.append("key_hex", req.body.key_hex as string);
    form.append("nonce_hex", req.body.nonce_hex as string);
    form.append("tag_hex", req.body.tag_hex as string);
    form.append("ciphertext_length", String(req.body.ciphertext_length));

    const response = await forwardJson("/extract", form);
    const buffer = Buffer.from(await response.arrayBuffer());
    const disposition = response.headers.get("content-disposition");
    if (disposition) {
      res.setHeader("Content-Disposition", disposition);
    }
    res
      .status(response.status)
      .setHeader("Content-Type", response.headers.get("content-type") || "application/octet-stream")
      .send(buffer);
  } catch (error) {
    res.status(400).json({ detail: String(error) });
  }
});

app.post("/detect", upload.single("image"), async (req, res) => {
  try {
    const image = requireFile(req.file, "image");
    const form = new FormData();
    appendFile(form, "image", image);
    const response = await forwardJson("/detect", form);
    const bodyText = await response.text();
    res.status(response.status).type("application/json").send(bodyText);
  } catch (error) {
    res.status(400).json({ detail: String(error) });
  }
});

app.listen(PORT, () => {
  console.log(`CipherNet TypeScript API listening on http://localhost:${PORT}`);
  console.log(`Proxying model requests to ${PYTHON_API_BASE_URL}`);
});
