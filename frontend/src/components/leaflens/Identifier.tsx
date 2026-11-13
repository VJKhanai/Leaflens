
import { useEffect, useRef, useState, type DragEvent } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import {
  AlertTriangle,
  BookOpen,
  Camera,
  ExternalLink,
  ImageUp,
  Info,
  Leaf,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  fetchPlantReport,
  predictLeaf,
  validateFile,
  type EdibilityClassification,
  type PlantReport,
  type PlantReportReference,
  type Prediction,
} from "@/lib/leaflens";

gsap.registerPlugin(useGSAP);

const EDIBILITY_TONES: Record<EdibilityClassification, string> = {
  Edible: "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
  "Not edible": "bg-destructive/10 text-destructive",
  "Conditionally edible": "bg-caution text-caution-foreground",
  "Unknown / Insufficient evidence": "bg-muted text-muted-foreground",
};

const SAFETY_TONES: Record<PlantReport["safety_assessment"]["category"], string> = {
  "Generally considered safe for documented uses":
    "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200",
  "Use with caution": "bg-caution text-caution-foreground",
  "Potentially harmful / Toxic": "bg-destructive/10 text-destructive",
  "Insufficient safety information": "bg-muted text-muted-foreground",
};

export function Identifier() {
  // References
  const root = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // States
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [plantReport, setPlantReport] = useState<PlantReport | null>(null);
  const [plantReportLoading, setPlantReportLoading] = useState(false);
  const [plantReportError, setPlantReportError] = useState<string | null>(null);
  const [plantReportQuery, setPlantReportQuery] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const reportRequest = useRef(0);
  const predictionRequest = useRef(0);

  const { contextSafe } = useGSAP({ scope: root });

  // Release image preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Stop camera when component is removed
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  // Stop camera
  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraActive(false);
  };

  // Select or remove an image
  const selectFile = (next: File | null | undefined) => {
    reportRequest.current += 1;
    predictionRequest.current += 1;
    setResult(null);
    setError(null);
    setPlantReport(null);
    setPlantReportError(null);
    setPlantReportLoading(false);
    setPlantReportQuery("");
    setLoading(false);

    if (inputRef.current) {
      inputRef.current.value = "";
    }

    if (!next) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    const problem = validateFile(next);

    if (problem) {
      setError(problem);
      return;
    }

    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
  };

  const loadPlantReport = async (name: string, requestId: number, refresh = false) => {
    setPlantReportQuery(name);
    setPlantReportLoading(true);
    setPlantReportError(null);
    setPlantReport(null);
    try {
      const report = await fetchPlantReport(name, refresh);
      if (requestId === reportRequest.current) {
        setPlantReport(report);
      }
    } catch (err) {
      if (requestId === reportRequest.current) {
        setPlantReportError(
          err instanceof Error ? err.message : "Unable to retrieve plant information."
        );
      }
    } finally {
      if (requestId === reportRequest.current) {
        setPlantReportLoading(false);
      }
    }
  };

  // Start camera
  const startCamera = async () => {
    try {
      setCameraError(null);

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not supported by this browser.");
      }

      stopCamera();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
        },
        audio: false,
      });

      streamRef.current = stream;
      setCameraActive(true);
    } catch (err) {
      setCameraError(
        err instanceof Error
          ? err.message
          : "Unable to access camera. Check camera permissions."
      );
    }
  };

  // Attach camera stream to video preview
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;

      videoRef.current.play().catch(() => {
        setCameraError("Unable to start camera preview.");
      });
    }
  }, [cameraActive]);

  // Capture photo
  const capturePhoto = () => {
    const video = videoRef.current;

    if (!video || !video.videoWidth || !video.videoHeight) {
      setCameraError("Camera is not ready. Please wait and try again.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      setCameraError("Unable to capture image.");
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setCameraError("Photo capture failed. Please try again.");
          return;
        }

        const photo = new File(
          [blob],
          `leaflens-camera-${Date.now()}.jpg`,
          { type: "image/jpeg" }
        );

        stopCamera();
        selectFile(photo);
      },
      "image/jpeg",
      0.92
    );
  };

  // Image preview animation
  useGSAP(
    () => {
      if (!previewUrl) return;

      gsap.fromTo(
        ".js-preview",
        {
          scale: 0.92,
          opacity: 0,
          clipPath: "inset(10% 10% 10% 10% round 18px)",
        },
        {
          scale: 1,
          opacity: 1,
          clipPath: "inset(0% 0% 0% 0% round 18px)",
          duration: 0.8,
          ease: "expo.out",
        }
      );
    },
    { scope: root, dependencies: [previewUrl] }
  );

  // Scanning animation
  useGSAP(
    () => {
      if (!loading) return;

      gsap.fromTo(
        ".js-scan",
        { yPercent: -100 },
        {
          yPercent: 100,
          duration: 1.4,
          ease: "sine.inOut",
          repeat: -1,
          yoyo: true,
        }
      );
    },
    { scope: root, dependencies: [loading], revertOnUpdate: true }
  );

  // Result reveal animation
  useGSAP(
    () => {
      if (!result || !resultRef.current) return;

      gsap.from(resultRef.current.querySelectorAll(".js-reveal"), {
        y: 24,
        opacity: 0,
        duration: 0.7,
        stagger: 0.09,
        ease: "power3.out",
      });

      resultRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    },
    { scope: root, dependencies: [result] }
  );

  // Identify leaf
  const onIdentify = contextSafe(async () => {
    if (!file) return;
    const requestId = ++predictionRequest.current;

    gsap.fromTo(
      ".js-identify",
      { scale: 0.96 },
      { scale: 1, duration: 0.5, ease: "back.out(3)" }
    );

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictLeaf(file);
      if (requestId !== predictionRequest.current) return;
      setResult(prediction);
      const scientificName = prediction.scientific_name;
      const hasVerifiedScientificName =
        scientificName &&
        scientificName !== "Not available" &&
        !/unconfirmed|unknown|possibly|not verified/i.test(scientificName);
      const lookupName = hasVerifiedScientificName
        ? scientificName
        : prediction.common_name !== "Not available"
          ? prediction.common_name
          : prediction.name;
      void loadPlantReport(lookupName, ++reportRequest.current);
    } catch (err) {
      if (requestId === predictionRequest.current) {
        setError(err instanceof Error ? err.message : "Prediction failed.");
      }
    } finally {
      if (requestId === predictionRequest.current) {
        setLoading(false);
      }
    }
  });

  // Drag and drop
  const onDrag = (e: DragEvent, over: boolean) => {
    e.preventDefault();
    setDragging(over);
  };

  return (
    <div ref={root} className="flex flex-col gap-8">

      {/* STEP 01 - UPLOAD AND CAMERA */}

      <div className="js-card rounded-3xl border border-border bg-card p-5 shadow-soft sm:p-7">

        <div className="mb-6 flex items-center justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
              Step 01
            </p>

            <h3 className="text-2xl font-medium">
              Choose your leaf photo
            </h3>
          </div>

          <span className="rounded-full bg-secondary px-3 py-1 text-xs font-semibold text-secondary-foreground">
            JPG · PNG · BMP
          </span>
        </div>

        {/* TWO EQUAL COLUMNS */}

        <div className="grid gap-6 md:grid-cols-2">

          {/* LEFT - UPLOAD */}

          <div className="min-w-0">

            <div className="mb-3 flex items-center gap-2">
              <ImageUp className="size-5 text-primary" />
              <h4 className="font-semibold">Upload from device</h4>
            </div>

            <div
              onDragEnter={(e) => onDrag(e, true)}
              onDragOver={(e) => onDrag(e, true)}
              onDragLeave={(e) => onDrag(e, false)}
              onDrop={(e) => {
                onDrag(e, false);
                selectFile(e.dataTransfer.files[0]);
              }}
              className={`dropzone-border relative flex min-h-[300px] flex-col justify-center overflow-hidden rounded-[18px] transition-colors ${
                dragging ? "bg-secondary" : "bg-muted/50"
              }`}
            >

              {previewUrl ? (

                <div className="js-preview relative p-3">

                  <div className="relative overflow-hidden rounded-xl">
                    <img
                      src={previewUrl}
                      alt="Selected leaf"
                      className="h-56 w-full rounded-xl object-cover"
                    />

                    {loading && (
                      <div className="pointer-events-none absolute inset-0 bg-forest/30">
                        <div className="js-scan scanline absolute inset-x-0 h-1/3" />
                      </div>
                    )}
                  </div>

                  <div className="mt-3 flex items-center justify-between gap-2">

                    <p className="truncate text-sm text-muted-foreground">
                      {file?.name}
                    </p>

                    <button
                      type="button"
                      onClick={() => selectFile(null)}
                      className="grid size-8 shrink-0 place-items-center rounded-full border border-border bg-card"
                      aria-label="Remove image"
                    >
                      <X className="size-4" />
                    </button>

                  </div>
                </div>

              ) : (

                <label
                  htmlFor="leaf-input"
                  className="flex min-h-[300px] cursor-pointer flex-col items-center justify-center px-5 text-center"
                >

                  <span className="mb-4 grid size-16 place-items-center rounded-2xl bg-card text-primary shadow-soft">
                    <ImageUp className="size-7" />
                  </span>

                  <span className="font-display text-xl">
                    Choose a leaf image
                  </span>

                  <span className="mt-2 text-sm text-muted-foreground">
                    Browse files or drag and drop here
                  </span>

                  <span className="mt-1 text-xs text-muted-foreground">
                    Maximum size: 10 MB
                  </span>

                </label>

              )}

              <input
                ref={inputRef}
                id="leaf-input"
                type="file"
                className="sr-only"
                accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
                onChange={(e) => selectFile(e.target.files?.[0])}
              />

            </div>
          </div>

          {/* RIGHT - CAMERA */}

          <div className="min-w-0">

            <div className="mb-3 flex items-center gap-2">
              <Camera className="size-5 text-primary" />
              <h4 className="font-semibold">Capture with camera</h4>
            </div>

            <div className="flex min-h-[300px] flex-col items-center justify-center overflow-hidden rounded-[18px] border border-border bg-muted/50 p-3">

              {cameraActive ? (

                <div className="w-full">

                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="h-56 w-full rounded-xl bg-black object-cover"
                  />

                  <div className="mt-4 flex w-full gap-3">

                    <Button
                      variant="hero"
                      className="flex-1"
                      onClick={capturePhoto}
                    >
                      <Camera className="size-4" />
                      Capture photo
                    </Button>

                    <Button
                      variant="outline"
                      onClick={stopCamera}
                    >
                      <X className="size-4" />
                      Stop
                    </Button>

                  </div>

                </div>

              ) : (

                <div className="flex flex-col items-center text-center">

                  <span className="mb-4 grid size-16 place-items-center rounded-2xl bg-card text-primary shadow-soft">
                    <Camera className="size-7" />
                  </span>

                  <h4 className="font-display text-xl">
                    Take a leaf photo
                  </h4>

                  <p className="mt-2 max-w-xs text-sm text-muted-foreground">
                    Use your device camera to capture a fresh leaf image.
                  </p>

                  <Button
                    variant="hero"
                    className="mt-5"
                    onClick={startCamera}
                  >
                    <Camera className="size-4" />
                    Open camera
                  </Button>

                </div>

              )}

              {cameraError && (
                <p className="mt-3 text-center text-sm text-destructive">
                  {cameraError}
                </p>
              )}

            </div>
          </div>

        </div>

        {/* COMMON IDENTIFY BUTTON */}

        <Button
          variant="hero"
          size="xl"
          className="js-identify mt-6 w-full"
          disabled={!file || loading}
          onClick={onIdentify}
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" />
              Analyzing leaf…
            </>
          ) : (
            <>
              <Sparkles />
              Identify leaf
            </>
          )}
        </Button>

        {error && (
          <p
            role="status"
            className="mt-4 flex items-center gap-2 rounded-xl bg-destructive/10 px-4 py-3 text-sm text-destructive"
          >
            <AlertTriangle className="size-4 shrink-0" />
            {error}
          </p>
        )}

      </div>

      {/* STEP 02 - PLANT REPORT */}

      <div
        ref={resultRef}
        className="js-card w-full overflow-hidden rounded-3xl border border-border bg-card p-5 text-foreground shadow-soft sm:p-8"
      >

        {result ? (

          <div className="space-y-7">

            {/* PLANT HEADING */}

            <div className="js-reveal flex flex-col gap-5 sm:flex-row sm:items-center">

              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Identified leaf"
                  className="h-36 w-36 shrink-0 rounded-2xl object-cover"
                />
              )}

              <div className="min-w-0">

                <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                  LeafLens AI · Plant Report
                </p>

                <h3 className="mt-2 break-words text-3xl font-semibold">
                  {result.name}
                </h3>

                <p className="mt-1 italic text-muted-foreground">
                  Scientific name not provided by the classifier
                </p>

                <p className="mt-3 font-semibold text-primary">
                  ViT model confidence: {result.score.toFixed(2)}%
                </p>

                <p className="mt-1 text-sm text-muted-foreground">
                  Leaf-image prediction · 80 classes
                </p>

              </div>
            </div>

            <div className="js-reveal border-t border-border" />

            <section className="js-reveal rounded-2xl border border-border bg-muted/40 p-4">
              <div className="flex items-center justify-between gap-3">
                <h4 className="text-lg font-semibold">Top 5 predictions</h4>
                <span className="text-xs text-muted-foreground">Confidence</span>
              </div>
              <ol className="mt-3 space-y-2">
                {result.predictions.map((prediction, index) => (
                  <li
                    key={prediction.class_index}
                    className="flex items-center justify-between gap-4 rounded-xl bg-card px-3 py-2 text-sm"
                  >
                    <span className="min-w-0 truncate">
                      <span className="mr-2 font-mono text-muted-foreground">{index + 1}.</span>
                      {prediction.class_name}
                    </span>
                    <span className="shrink-0 font-mono font-semibold">
                      {prediction.confidence_percent.toFixed(2)}%
                    </span>
                  </li>
                ))}
              </ol>
              <p className="mt-3 text-xs text-muted-foreground">
                Predictions are based on leaf images and do not verify species identity or plant safety.
              </p>
            </section>

            <section className="js-reveal space-y-4" aria-labelledby="research-report-title">
              <div>
                <h4 id="research-report-title" className="flex items-center gap-2 text-xl font-semibold">
                  <BookOpen className="size-5 text-primary" />
                  Botanical research report
                </h4>
                <p className="mt-1 text-sm text-muted-foreground">
                  Source-linked taxonomy and literature. Missing data is shown as unverified, not inferred.
                </p>
              </div>

              {plantReportLoading && (
                <p role="status" className="flex items-center gap-2 rounded-xl bg-muted px-4 py-3 text-sm">
                  <Loader2 className="size-4 animate-spin" />
                  Retrieving botanical sources…
                </p>
              )}

              {plantReportError && (
                <div role="alert" className="rounded-xl border border-caution-border bg-caution p-4 text-sm text-caution-foreground">
                  <p>{plantReportError}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-3"
                    onClick={() => loadPlantReport(
                      plantReportQuery,
                      ++reportRequest.current,
                      true
                    )}
                  >
                    <RotateCcw className="size-4" />
                    Retry report
                  </Button>
                </div>
              )}

              {plantReport && (
                <div className="space-y-4">
                  <div className="rounded-2xl border border-border bg-muted/40 p-4">
                    <p className="flex items-start gap-2 text-sm">
                      <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                      {plantReport.basic_information.identity_warning}
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {plantReport.taxonomy
                        ? `GBIF species-match confidence: ${plantReport.taxonomy.confidence}%. This is not image-identification confidence.`
                        : "Species-level identity was not verified; species-specific information is unavailable."}
                    </p>
                  </div>

                  <Accordion
                    type="multiple"
                    defaultValue={["overview"]}
                    className="rounded-2xl border border-border px-4"
                  >
                    <AccordionItem value="overview">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <Leaf className="size-4 text-primary" />
                          Plant overview
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-4">
                        <div className="grid gap-4 sm:grid-cols-2">
                          <InformationValue label="Common name" value={plantReport.basic_information.common_name} />
                          <InformationValue label="Scientific name" value={plantReport.basic_information.scientific_name} />
                          <InformationValue label="Family" value={plantReport.basic_information.family} />
                          <InformationValue label="Native region / distribution" value={plantReport.basic_information.native_region} />
                        </div>
                        <InformationValue label="Botanical description" value={plantReport.basic_information.description} />
                        <div>
                          <h5 className="text-sm font-semibold">Other common or regional names</h5>
                          {plantReport.basic_information.other_names.length ? (
                            <ul className="mt-1 list-inside list-disc text-sm text-foreground/80">
                              {plantReport.basic_information.other_names.map(({ name, language }, index) => (
                                <li key={`${name}-${language ?? index}`}>
                                  {name}{language ? ` (${language})` : ""}
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <p className="mt-1 text-sm text-muted-foreground">No additional verified names available.</p>
                          )}
                        </div>
                        <ReferenceLinks
                          referenceIds={plantReport.basic_information.reference_ids}
                          references={plantReport.references}
                        />
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="edibility">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <ShieldCheck className="size-4 text-primary" />
                          Edibility &amp; food safety
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-4">
                        <span className={`inline-flex rounded-full px-3 py-1 text-sm font-semibold ${EDIBILITY_TONES[plantReport.edibility.classification]}`}>
                          {plantReport.edibility.classification}
                        </span>
                        <p className="text-sm leading-relaxed text-foreground/80">
                          {plantReport.edibility.summary}
                        </p>
                        <div className="grid gap-4 sm:grid-cols-2">
                          <InformationList label="Verified edible parts" items={plantReport.edibility.edible_parts} />
                          <InformationList label="Parts to avoid" items={plantReport.edibility.parts_to_avoid} />
                          <InformationValue label="Required preparation" value={plantReport.edibility.preparation} />
                          <InformationValue label="Toxicity concerns" value={plantReport.edibility.toxicity_concerns} />
                        </div>
                        <ReferenceLinks
                          referenceIds={plantReport.edibility.reference_ids}
                          references={plantReport.references}
                        />
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="medicinal">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <BookOpen className="size-4 text-primary" />
                          Medicinal benefits &amp; evidence
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-4">
                        <p className="text-sm leading-relaxed text-foreground/80">
                          {plantReport.medicinal_benefits.evidence_summary}
                        </p>
                        <InformationList label="Traditionally reported uses" items={plantReport.medicinal_benefits.traditional_uses} />
                        <InformationList label="Documented properties (not clinical advice)" items={plantReport.medicinal_benefits.documented_properties} />
                        <InformationList label="Parts traditionally used" items={plantReport.medicinal_benefits.parts_traditionally_used} />
                        {plantReport.medicinal_benefits.literature_search_results.map((paper) => (
                          <article key={paper.url} className="rounded-xl bg-muted/50 p-3">
                            <a
                              href={paper.url}
                              target="_blank"
                              rel="noreferrer"
                              className="font-medium text-primary underline"
                            >
                              {paper.title} <ExternalLink className="inline size-3" />
                            </a>
                            <p className="mt-1 text-xs text-muted-foreground">
                              {[paper.authors, paper.journal, paper.publication_date ?? paper.year]
                                .filter(Boolean)
                                .join(" · ")}
                            </p>
                            {paper.abstract && (
                              <p className="mt-2 line-clamp-5 whitespace-pre-line text-sm leading-relaxed">
                                {paper.abstract}
                              </p>
                            )}
                          </article>
                        ))}
                        <ReferenceLinks
                          referenceIds={plantReport.medicinal_benefits.reference_ids}
                          references={plantReport.references}
                        />
                        <p className="text-xs text-muted-foreground">
                          Literature search matches are not proof of efficacy; no treatment or dosage advice is provided.
                        </p>
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="disadvantages">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <AlertTriangle className="size-4 text-destructive" />
                          Disadvantages &amp; side effects
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-4">
                        <InformationList label="Known side effects" items={plantReport.disadvantages.known_side_effects} />
                        <div className="grid gap-4 sm:grid-cols-2">
                          <InformationValue label="Toxicity concerns" value={plantReport.disadvantages.toxicity_concerns} />
                          <InformationValue label="Possible allergic reactions" value={plantReport.disadvantages.allergic_reactions} />
                          <InformationValue label="Excessive consumption risks" value={plantReport.disadvantages.excessive_consumption_risks} />
                          <InformationValue label="Incorrect preparation risks" value={plantReport.disadvantages.preparation_risks} />
                          <InformationValue label="Documented medication interactions" value={plantReport.disadvantages.medication_interactions} />
                        </div>
                        <p className="text-sm text-muted-foreground">{plantReport.disadvantages.summary}</p>
                        <ReferenceLinks
                          referenceIds={plantReport.disadvantages.reference_ids}
                          references={plantReport.references}
                        />
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="uses">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <Sparkles className="size-4 text-primary" />
                          Documented uses
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-4">
                        <InformationList label="Traditional medicinal" items={plantReport.uses.traditional_medicinal} />
                        <InformationList label="Culinary" items={plantReport.uses.culinary} />
                        <InformationList label="Agricultural or gardening" items={plantReport.uses.agricultural_gardening} />
                        <InformationList label="Commercial or industrial" items={plantReport.uses.commercial_industrial} />
                        <InformationList label="Other documented applications" items={plantReport.uses.other_documented} />
                        <ReferenceLinks
                          referenceIds={plantReport.uses.reference_ids}
                          references={plantReport.references}
                        />
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="safety">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <ShieldCheck className="size-4 text-primary" />
                          Safety assessment
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="space-y-3">
                        <p className={`inline-flex rounded-full px-3 py-1 text-sm font-semibold ${SAFETY_TONES[plantReport.safety_assessment.category]}`}>
                          {plantReport.safety_assessment.category}
                        </p>
                        <p className="text-sm leading-relaxed text-foreground/80">
                          {plantReport.safety_assessment.summary}
                        </p>
                        <p className="flex items-start gap-2 rounded-xl bg-caution p-3 text-sm text-caution-foreground">
                          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                          {plantReport.safety_assessment.ingestion_warning}
                        </p>
                        <ReferenceLinks
                          referenceIds={plantReport.safety_assessment.reference_ids}
                          references={plantReport.references}
                        />
                      </AccordionContent>
                    </AccordionItem>

                    <AccordionItem value="references">
                      <AccordionTrigger>
                        <span className="flex items-center gap-2">
                          <ExternalLink className="size-4 text-primary" />
                          Scientific references
                        </span>
                      </AccordionTrigger>
                      <AccordionContent>
                        {plantReport.references.length ? (
                          <ul className="space-y-3">
                            {plantReport.references.map((reference) => (
                              <li key={reference.id} className="rounded-xl bg-muted/50 p-3 text-sm">
                                <a
                                  href={reference.url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="font-medium text-primary underline"
                                >
                                  {reference.title} <ExternalLink className="inline size-3" />
                                </a>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  {[reference.source, reference.author, reference.publication_date ?? reference.year]
                                    .filter(Boolean)
                                    .join(" · ")}
                                  {reference.accessed_at ? ` · Accessed ${reference.accessed_at}` : ""}
                                </p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Supports: {reference.supports.join(", ")}
                                </p>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-sm text-muted-foreground">No verified source records are available.</p>
                        )}
                        <div className="mt-4 flex flex-wrap gap-2 text-xs">
                          {plantReport.sources.map((source) => (
                            <span key={source.name} className="rounded-full bg-muted px-3 py-1">
                              {source.name}: {source.status.replaceAll("_", " ")}
                            </span>
                          ))}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </div>
              )}
            </section>

            <p className="js-reveal text-sm leading-relaxed text-muted-foreground">
              AI leaf classification is not a safety test. Verify the plant with a qualified botanist before any use or consumption.
            </p>

            {/* IDENTIFY ANOTHER LEAF */}

            <Button
              variant="hero"
              size="lg"
              className="js-reveal"
              onClick={() => selectFile(null)}
            >
              <RotateCcw />
              Identify another leaf
            </Button>

          </div>

        ) : (

          <div className="flex min-h-48 flex-col items-center justify-center text-center">

            <Leaf className="mb-4 size-12 text-primary" />

            <h3 className="text-2xl font-medium">
              Your plant report
            </h3>

            <p className="mt-2 max-w-md text-sm text-muted-foreground">
              Upload a leaf image or capture one using your camera to generate its botanical report.
            </p>

          </div>

        )}

      </div>

    </div>
  );
}


function InformationValue({ label, value }: { label: string; value: string | null }) {
  return (
    <div>
      <h5 className="text-sm font-semibold">{label}</h5>
      <p className="mt-1 whitespace-pre-line text-sm text-foreground/80">
        {value?.trim() || "Information not verified."}
      </p>
    </div>
  );
}

function InformationList({ label, items }: { label: string; items: string[] }) {
  return (
    <div>
      <h5 className="text-sm font-semibold">{label}</h5>
      {items.length ? (
        <ul className="mt-1 list-inside list-disc space-y-1 text-sm text-foreground/80">
          {items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
        </ul>
      ) : (
        <p className="mt-1 text-sm text-muted-foreground">Information not verified.</p>
      )}
    </div>
  );
}

function ReferenceLinks({
  referenceIds,
  references,
}: {
  referenceIds: string[];
  references: PlantReportReference[];
}) {
  const citedReferences = referenceIds
    .map((id) => references.find((reference) => reference.id === id))
    .filter((reference): reference is PlantReportReference => reference !== undefined);

  if (!citedReferences.length) return null;

  return (
    <div>
      <h5 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Sources for this information
      </h5>
      <ul className="mt-1 space-y-1 text-sm">
        {citedReferences.map((reference) => (
          <li key={reference.id}>
            <a
              href={reference.url}
              target="_blank"
              rel="noreferrer"
              className="text-primary underline"
            >
              {reference.source}: {reference.title} <ExternalLink className="inline size-3" />
            </a>
            {reference.accessed_at ? (
              <span className="text-xs text-muted-foreground"> · Accessed {reference.accessed_at}</span>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
