import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchPlantReport, predictLeaf, validateFile } from "@/lib/leaflens";

describe("leaf image validation", () => {
  it("accepts a JPG under 10 MB", () => {
    expect(validateFile({ type: "image/jpeg", size: 2 * 1024 * 1024 })).toBeNull();
  });

  describe("ViT identification API", () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("uses the ViT label and preserves top-five and 80-class output", async () => {
      const all_predictions = Array.from({ length: 80 }, (_, class_index) => ({
        class_index,
        class_name: `Plant ${class_index}`,
        confidence_percent: 100 - class_index,
      }));
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          prediction: "Neem",
          confidence: 100,
          predictions: all_predictions.slice(0, 5),
          all_predictions,
          class_count: 80,
          model: "dhivyasri-d/vit-plant-classifier",
        }),
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await predictLeaf(new File(["image"], "leaf.png", { type: "image/png" }));

      expect(result.name).toBe("Plant 0");
      expect(result.predictions).toHaveLength(5);
      expect(result.all_predictions).toHaveLength(80);
      expect(result.model).toBe("dhivyasri-d/vit-plant-classifier");
    });
  });

  describe("plant information reports", () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("requests a report using the encoded plant name", async () => {
      const report = { query: "Azadirachta indica", status: "partial" };
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => report,
      });
      vi.stubGlobal("fetch", fetchMock);

      await expect(fetchPlantReport("Azadirachta indica")).resolves.toEqual(report);
      expect(fetchMock).toHaveBeenCalledWith(
        `${window.location.origin}/plant-info?name=Azadirachta+indica`
      );
    });

    it("bypasses the API cache when retrying a report", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ query: "Neem", status: "unavailable" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await fetchPlantReport("Neem", true);

      expect(fetchMock).toHaveBeenCalledWith(
        `${window.location.origin}/plant-info?name=Neem&refresh=1`
      );
    });

    it("surfaces report service failures for retry in the UI", async () => {
      vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

      await expect(fetchPlantReport("Neem")).rejects.toThrow(
        "Couldn't reach the botanical information service. Please retry."
      );
    });
  });
  it("rejects files larger than 10 MB", () => {
    expect(validateFile({ type: "image/png", size: 10 * 1024 * 1024 + 1 })).toBe("Image exceeds the 10 MB limit.");
  });
  it("rejects unsupported formats like WEBP", () => {
    expect(validateFile({ type: "image/webp", size: 1000 })).toBe("Choose a JPG, PNG, or BMP image.");
  });
});
