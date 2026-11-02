export const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/bmp"]);
export const MAX_BYTES = 10 * 1024 * 1024;

export type Prediction = {
  name: string;
  score: number;
  uses: string;
  edibility: string;
  safety: string;
  warning: string;
};

/** Returns an error message for an invalid file, or null when it is acceptable. */
export function validateFile(file: { type: string; size: number }): string | null {
  if (!ALLOWED_TYPES.has(file.type)) return "Choose a JPG, PNG, or BMP image.";
  if (file.size > MAX_BYTES) return "Image exceeds the 10 MB limit.";
  return null;
}

// The prediction model runs on a separate server. Point VITE_PREDICT_URL at it;
// defaults to the same-origin /predict endpoint the original page used.
const PREDICT_URL = (import.meta.env["VITE_PREDICT_URL"] as string | undefined) || "/predict";

export async function predictLeaf(file: File): Promise<Prediction> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(PREDICT_URL, { method: "POST", body: formData });
  } catch {
    throw new Error("Couldn't reach the identification service. Please try again.");
  }

  type ApiResponse = {
    error?: string;
    prediction?: string;
    confidence?: number | string;
    warning?: string;
    plant_info?: { uses?: string; edibility?: string; safety?: string };
  };
  let data: ApiResponse = {};
  try {
    data = await response.json();
  } catch {
    throw new Error("The identification service isn't available right now.");
  }

  if (!response.ok) {
    throw new Error(data.error || "Prediction failed.");
  }

  const info = data.plant_info ?? {};
  return {
    name: String(data.prediction ?? "Unknown species"),
    score: Math.min(100, Math.max(0, Number(data.confidence) || 0)),
    uses: info.uses || "Uses have not been verified.",
    edibility: info.edibility || "UNKNOWN — do not eat this plant based on an AI prediction.",
    safety: info.safety || "Verify the species with a reliable botanical source.",
    warning: data.warning || "AI prediction only. Verify independently.",
  };
}
