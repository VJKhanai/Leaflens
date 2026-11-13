
export const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/bmp"]);
export const MAX_BYTES = 10 * 1024 * 1024;

export type Prediction = {
  name: string;
  score: number;
  common_name: string;
  scientific_name: string;
  description: string;
  benefits: string[];
  disadvantages: string[];
  proper_usage: string;
  applications: string[];
  edibility: string;
  harmful_compounds: string;
  if_consumed: string;
  precautions: string;
  uses: string;
  safety: string;
  warning: string;
};

export function validateFile(file: { type: string; size: number }): string | null {
  if (!ALLOWED_TYPES.has(file.type)) return "Choose a JPG, PNG, or BMP image.";
  if (file.size > MAX_BYTES) return "Image exceeds the 10 MB limit.";
  return null;
}

const PREDICT_URL =
  (import.meta.env["VITE_PREDICT_URL"] as string | undefined) || "/predict";

export async function predictLeaf(file: File): Promise<Prediction> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;

  try {
    response = await fetch(PREDICT_URL, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Couldn't reach the identification service. Please try again.");
  }

  type PlantInfo = {
    common_name?: string;
    scientific_name?: string;
    description?: string;
    benefits?: string[];
    disadvantages?: string[];
    proper_usage?: string;
    applications?: string[];
    edibility?: string;
    harmful_compounds?: string;
    if_consumed?: string;
    precautions?: string;
    uses?: string;
    safety?: string;
  };

  type ApiResponse = {
    error?: string;
    prediction?: string;
    confidence?: number | string;
    warning?: string;
    plant_info?: PlantInfo;
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

    common_name: info.common_name || "Not available",
    scientific_name: info.scientific_name || "Not available",
    description: info.description || "Description not available.",
    benefits: info.benefits || [],
    disadvantages: info.disadvantages || [],
    proper_usage: info.proper_usage || "Usage information not available.",
    applications: info.applications || [],
    edibility: info.edibility || "UNKNOWN — do not eat based on an AI prediction.",
    harmful_compounds: info.harmful_compounds || "Information not available.",
    if_consumed: info.if_consumed || "Seek professional medical or poison-control advice if ingestion is suspected.",
    precautions: info.precautions || "Verify the species with a reliable botanical expert.",

    uses: info.uses || "Uses have not been verified.",
    safety: info.safety || info.precautions || "Verify the species with a reliable botanical source.",
    warning: data.warning || "AI prediction only. Verify independently.",
  };
}
