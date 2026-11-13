
import { useEffect, useRef, useState, type DragEvent } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import {
  AlertTriangle,
  Camera,
  ImageUp,
  Leaf,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { predictLeaf, validateFile, type Prediction } from "@/lib/leaflens";

gsap.registerPlugin(useGSAP);

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
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

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
    setResult(null);
    setError(null);

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
      setResult(prediction);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Prediction failed."
      );
    } finally {
      setLoading(false);
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
                  {result.common_name}
                </h3>

                <p className="mt-1 italic text-muted-foreground">
                  {result.scientific_name}
                </p>

                <p className="mt-3 font-semibold text-primary">
                  Model score: {result.score.toFixed(2)}%
                </p>

                <p className="mt-1 text-sm text-muted-foreground">
                  Predicted species: {result.name}
                </p>

              </div>
            </div>

            <div className="js-reveal border-t border-border" />

            {/* BENEFITS AND APPLICATIONS */}

            <div className="grid gap-x-10 gap-y-8 sm:grid-cols-2">

              <ReportSection
                icon={<Leaf className="size-5 text-primary" />}
                title="Benefits"
                items={result.benefits}
              />

              <ReportSection
                icon={<AlertTriangle className="size-5 text-destructive" />}
                title="Disadvantages"
                items={result.disadvantages}
              />

              <ReportSection
                icon={<Leaf className="size-5 text-primary" />}
                title="Proper usage"
                text={result.proper_usage}
              />

              <ReportSection
                icon={<Leaf className="size-5 text-primary" />}
                title="Where it's used"
                items={result.applications}
              />

            </div>

            {/* DESCRIPTION */}

            <ReportSection
              icon={<Leaf className="size-5 text-primary" />}
              title="Plant description"
              text={result.description}
            />

            {/* SAFETY INFORMATION */}

            <div className="js-reveal space-y-5 border-t border-border pt-6">

              <ReportSection
                icon={<ShieldCheck className="size-5 text-primary" />}
                title="Edibility"
                text={result.edibility}
              />

              <ReportSection
                icon={<AlertTriangle className="size-5 text-destructive" />}
                title="Harmful compounds"
                text={result.harmful_compounds}
              />

              <ReportSection
                icon={<AlertTriangle className="size-5 text-destructive" />}
                title="What happens if consumed?"
                text={result.if_consumed}
              />

              <ReportSection
                icon={<ShieldCheck className="size-5 text-primary" />}
                title="Safety precautions"
                text={result.precautions}
              />

            </div>

            {/* WARNING */}

            <p className="js-reveal text-sm leading-relaxed text-muted-foreground">
              {result.warning} Plant identification must be independently
              verified before any use or consumption.
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


// REUSABLE PLANT INFORMATION SECTION

function ReportSection({
  icon,
  title,
  text,
  items,
}: {
  icon: React.ReactNode;
  title: string;
  text?: string;
  items?: string[];
}) {
  return (
    <div className="js-reveal space-y-2">

      <h4 className="flex items-center gap-3 text-lg font-semibold">
        {icon}
        {title}
      </h4>

      {text && (
        <p className="whitespace-pre-line text-base leading-relaxed text-foreground/80">
          {text}
        </p>
      )}

      {items && (
        <ul className="space-y-1 text-base leading-relaxed text-foreground/80">
          {items.length > 0 ? (
            items.map((item, index) => (
              <li key={index}>• {item}</li>
            ))
          ) : (
            <li>Information not available.</li>
          )}
        </ul>
      )}

    </div>
  );
}
