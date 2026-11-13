
export const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/bmp"]);
export const MAX_BYTES = 10 * 1024 * 1024;

export type PredictionCandidate = {
  class_index: number;
  class_name: string;
  confidence_percent: number;
};

export type Prediction = {
  name: string;
  score: number;
  predictions: PredictionCandidate[];
  all_predictions: PredictionCandidate[];
  model: string;
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

export type PlantReportSection = {
  status: string;
  message: string;
  papers?: Array<{
    title: string;
    authors?: string;
    year?: string | number | null;
    publication_date?: string | number | null;
    journal?: string;
    url: string;
    abstract?: string;
  }>;
};

export type PlantReportReference = {
  id: string;
  source: string;
  title: string;
  author?: string | null;
  year?: string | number | null;
  publication_date?: string | number | null;
  accessed_at?: string | null;
  url: string;
  supports: string[];
};

export type EdibilityClassification =
  | "Edible"
  | "Not edible"
  | "Conditionally edible"
  | "Unknown / Insufficient evidence";

export type PlantReport = {
  query: string;
  status: "partial" | "unavailable";
  taxonomy: {
    scientific_name: string;
    canonical_name?: string;
    rank?: string;
    family?: string;
    genus?: string;
    accepted_name?: string;
    confidence: number;
    source_url: string;
  } | null;
  basic_information: {
    common_name: string | null;
    scientific_name: string | null;
    other_names: Array<{ name: string; language: string | null }>;
    family: string | null;
    description: string | null;
    native_region: string | null;
    taxonomy_confidence: number | null;
    identity_warning: string;
    reference_ids: string[];
  };
  edibility: {
    classification: EdibilityClassification;
    edible_parts: string[];
    parts_to_avoid: string[];
    preparation: string | null;
    toxicity_concerns: string | null;
    summary: string;
    evidence_status: string;
    reference_ids: string[];
  };
  medicinal_benefits: {
    traditional_uses: string[];
    documented_properties: string[];
    parts_traditionally_used: string[];
    evidence_summary: string;
    clinical_evidence_status: string;
    literature_search_results: Array<{
      title: string;
      authors?: string;
      year?: string | number | null;
      publication_date?: string | number | null;
      journal?: string;
      url: string;
      abstract?: string;
    }>;
    reference_ids: string[];
  };
  disadvantages: {
    known_side_effects: string[];
    toxicity_concerns: string | null;
    allergic_reactions: string | null;
    excessive_consumption_risks: string | null;
    preparation_risks: string | null;
    medication_interactions: string | null;
    summary: string;
    evidence_status: string;
    reference_ids: string[];
  };
  uses: {
    traditional_medicinal: string[];
    culinary: string[];
    agricultural_gardening: string[];
    commercial_industrial: string[];
    other_documented: string[];
    evidence_status: string;
    reference_ids: string[];
  };
  safety_assessment: {
    category:
      | "Generally considered safe for documented uses"
      | "Use with caution"
      | "Potentially harmful / Toxic"
      | "Insufficient safety information";
    summary: string;
    ingestion_warning: string;
    reference_ids: string[];
  };
  overview: {
    status: string;
    common_names: string[];
    message: string;
  };
  sections: {
    edibility: PlantReportSection;
    nutrition: PlantReportSection;
    benefits: PlantReportSection;
    risks: PlantReportSection;
    medicinal_uses: PlantReportSection;
    agricultural_uses: PlantReportSection;
    commercial_uses: PlantReportSection;
    cultivation: PlantReportSection;
    safety_summary: PlantReportSection;
  };
  references: PlantReportReference[];
  sources: Array<{
    name: string;
    status: string;
    url?: string;
    message?: string;
  }>;
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
    predictions?: unknown;
    all_predictions?: unknown;
    class_count?: number;
    model?: string;
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

  const predictions = Array.isArray(data.predictions)
    ? data.predictions.filter(isPredictionCandidate)
    : [];
  const allPredictions = Array.isArray(data.all_predictions)
    ? data.all_predictions.filter(isPredictionCandidate)
    : [];
  const topPrediction = predictions[0];
  if (
    !Array.isArray(data.predictions) ||
    data.predictions.length !== 5 ||
    predictions.length !== 5 ||
    !Array.isArray(data.all_predictions) ||
    data.all_predictions.length !== 80 ||
    allPredictions.length !== 80 ||
    data.class_count !== 80 ||
    !topPrediction
  ) {
    throw new Error("The identification service returned incomplete ViT predictions.");
  }

  const info = data.plant_info ?? {};

  return {
    name: topPrediction.class_name,
    score: topPrediction.confidence_percent,
    predictions,
    all_predictions: allPredictions,
    model: data.model || "dhivyasri-d/vit-plant-classifier",

    common_name: info.common_name || "Not available",
    scientific_name: info.scientific_name || "Not available",
    description: info.description || "No verified plant description was supplied by the classifier.",
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

function isPredictionCandidate(value: unknown): value is PredictionCandidate {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate["class_index"] === "number" &&
    Number.isInteger(candidate["class_index"]) &&
    candidate["class_index"] >= 0 &&
    candidate["class_index"] < 80 &&
    typeof candidate["class_name"] === "string" &&
    candidate["class_name"].length > 0 &&
    typeof candidate["confidence_percent"] === "number" &&
    Number.isFinite(candidate["confidence_percent"]) &&
    candidate["confidence_percent"] >= 0 &&
    candidate["confidence_percent"] <= 100
  );
}

const PLANT_INFO_URL =
  (import.meta.env["VITE_PLANT_INFO_URL"] as string | undefined) || "/plant-info";

export async function fetchPlantReport(name: string, refresh = false): Promise<PlantReport> {
  const url = new URL(PLANT_INFO_URL, window.location.origin);
  url.searchParams.set("name", name);
  if (refresh) url.searchParams.set("refresh", "1");

  let response: Response;
  try {
    response = await fetch(url.toString());
  } catch {
    throw new Error("Couldn't reach the botanical information service. Please retry.");
  }

  let data: PlantReport | { error?: string };
  try {
    data = await response.json();
  } catch {
    throw new Error("The botanical information service returned an invalid response.");
  }

  if (!response.ok) {
    throw new Error("error" in data ? data.error || "Plant report retrieval failed." : "Plant report retrieval failed.");
  }

  return data as PlantReport;
}
