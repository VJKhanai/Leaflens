import { describe, expect, it } from "vitest";
import { validateFile } from "@/lib/leaflens";

describe("leaf image validation", () => {
  it("accepts a JPG under 10 MB", () => {
    expect(validateFile({ type: "image/jpeg", size: 2 * 1024 * 1024 })).toBeNull();
  });
  it("rejects files larger than 10 MB", () => {
    expect(validateFile({ type: "image/png", size: 10 * 1024 * 1024 + 1 })).toBe("Image exceeds the 10 MB limit.");
  });
  it("rejects unsupported formats like WEBP", () => {
    expect(validateFile({ type: "image/webp", size: 1000 })).toBe("Choose a JPG, PNG, or BMP image.");
  });
});
