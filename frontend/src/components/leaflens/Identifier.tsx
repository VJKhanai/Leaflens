import { useEffect, useRef, useState, type DragEvent } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { AlertTriangle, ImageUp, Leaf, Loader2, RotateCcw, ShieldCheck, Sparkles, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { predictLeaf, validateFile, type Prediction } from "@/lib/leaflens";

gsap.registerPlugin(useGSAP);

export function Identifier() {
  const root = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const scoreRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const { contextSafe } = useGSAP({ scope: root });

  const selectFile = (next: File | null | undefined) => {
    setResult(null);
    setError(null);
    setFile(null);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
    if (!next) return;
    const problem = validateFile(next);
    if (problem) {
      setError(problem);
      gsap.fromTo(".js-error", { x: -6 }, { x: 0, duration: 0.5, ease: "elastic.out(1, 0.3)" });
      return;
    }
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
  };

  // Preview pop-in
  useGSAP(
    () => {
      if (!previewUrl) return;
      gsap.fromTo(
        ".js-preview",
        { scale: 0.92, opacity: 0, clipPath: "inset(10% 10% 10% 10% round 18px)" },
        { scale: 1, opacity: 1, clipPath: "inset(0% 0% 0% 0% round 18px)", duration: 0.8, ease: "expo.out" },
      );
    },
    { scope: root, dependencies: [previewUrl] },
  );

  // Scanning loop while analysing
  useGSAP(
    () => {
      if (!loading) return;
      gsap.fromTo(".js-scan", { yPercent: -100 }, { yPercent: 100, duration: 1.4, ease: "sine.inOut", repeat: -1, yoyo: true });
    },
    { scope: root, dependencies: [loading], revertOnUpdate: true },
  );

  // Result reveal
  useGSAP(
    () => {
      if (!result || !resultRef.current) return;
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
      tl.from(resultRef.current.querySelectorAll(".js-reveal"), { y: 24, opacity: 0, duration: 0.7, stagger: 0.09 });
      const counter = { v: 0 };
      tl.to(
        counter,
        {
          v: result.score,
          duration: 1.4,
          ease: "power2.out",
          onUpdate: () => {
            if (scoreRef.current) scoreRef.current.textContent = counter.v.toFixed(2);
          },
        },
        0.2,
      );
      tl.fromTo(barRef.current, { width: "0%" }, { width: `${result.score}%`, duration: 1.4, ease: "power2.out" }, 0.2);
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    },
    { scope: root, dependencies: [result] },
  );

  const onIdentify = contextSafe(async () => {
    if (!file) return;
    gsap.fromTo(".js-identify", { scale: 0.96 }, { scale: 1, duration: 0.5, ease: "back.out(3)" });
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predictLeaf(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  });

  const onDrag = (e: DragEvent, over: boolean) => {
    e.preventDefault();
    setDragging(over);
  };

  return (
    <div ref={root} className="grid gap-6 lg:grid-cols-[1.05fr_1fr]">
      {/* Upload card */}
      <div className="js-card rounded-3xl border border-border bg-card p-5 shadow-soft sm:p-7">
        <div className="mb-5 flex items-center justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Step 01</p>
            <h3 className="text-2xl font-medium">Upload a leaf</h3>
          </div>
          <span className="rounded-full bg-secondary px-3 py-1 text-xs font-semibold text-secondary-foreground">
            JPG · PNG · BMP
          </span>
        </div>

        <div
          onDragEnter={(e) => onDrag(e, true)}
          onDragOver={(e) => onDrag(e, true)}
          onDragLeave={(e) => onDrag(e, false)}
          onDrop={(e) => {
            onDrag(e, false);
            selectFile(e.dataTransfer.files[0]);
          }}
          className={cn(
            "dropzone-border relative overflow-hidden rounded-[18px] transition-colors duration-300",
            dragging ? "bg-secondary" : "bg-muted/50",
          )}
        >
          {previewUrl ? (
            <div className="js-preview relative p-3">
              <div className="relative overflow-hidden rounded-xl">
                <img src={previewUrl} alt="Selected leaf" className="mx-auto max-h-80 w-full rounded-xl object-cover" />
                {loading && (
                  <div className="pointer-events-none absolute inset-0 bg-forest/30">
                    <div className="js-scan scanline absolute inset-x-0 h-1/3" />
                  </div>
                )}
              </div>
              <div className="mt-3 flex items-center justify-between gap-3 px-1">
                <p className="truncate text-sm text-muted-foreground">{file?.name}</p>
                <button
                  type="button"
                  onClick={() => selectFile(null)}
                  className="grid size-8 shrink-0 place-items-center rounded-full border border-border bg-card text-muted-foreground transition hover:text-foreground"
                  aria-label="Remove image"
                >
                  <X className="size-4" />
                </button>
              </div>
            </div>
          ) : (
            <label
              htmlFor="leaf-input"
              className="flex cursor-pointer flex-col items-center px-6 py-14 text-center"
            >
              <span
                className={cn(
                  "mb-5 grid size-16 place-items-center rounded-2xl bg-card text-primary shadow-soft transition-transform duration-500",
                  dragging && "-translate-y-1 scale-110",
                )}
              >
                <ImageUp className="size-7" />
              </span>
              <span className="font-display text-xl">Drop your photo here</span>
              <span className="mt-1 text-sm text-muted-foreground">
                or <span className="font-semibold text-primary underline underline-offset-4">browse files</span> · max 10 MB
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

        <Button
          variant="hero"
          size="xl"
          className="js-identify mt-5 w-full"
          disabled={!file || loading}
          onClick={onIdentify}
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" /> Analyzing leaf…
            </>
          ) : (
            <>
              <Sparkles /> Identify leaf
            </>
          )}
        </Button>

        {error && (
          <p role="status" className="js-error mt-4 flex items-center gap-2 rounded-xl bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <AlertTriangle className="size-4 shrink-0" /> {error}
          </p>
        )}
      </div>

      {/* Result card */}
      <div ref={resultRef} className="js-card relative overflow-hidden rounded-3xl bg-forest-gradient p-5 text-forest-foreground shadow-lift sm:p-7">
        <div className="pointer-events-none absolute -right-24 -top-24 size-72 rounded-full bg-primary-glow/10 blur-3xl" />
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-forest-foreground/60">Step 02</p>
        <h3 className="text-2xl font-medium">Field report</h3>

        {result ? (
          <div className="relative mt-6 space-y-4">
            <div className="js-reveal">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary-glow">Predicted species</p>
              <h4 className="mt-1 break-words font-display text-4xl font-medium italic leading-tight sm:text-5xl">
                {result.name}
              </h4>
            </div>

            <div className="js-reveal rounded-2xl bg-forest-foreground/5 p-4 ring-1 ring-forest-foreground/10">
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-forest-foreground/70">Model score</span>
                <span className="font-mono text-2xl font-semibold text-primary-glow">
                  <span ref={scoreRef}>0.00</span>%
                </span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-forest-foreground/10">
                <div ref={barRef} className="h-full w-0 rounded-full bg-glow-gradient" />
              </div>
            </div>

            <InfoBlock icon={<Leaf className="size-4" />} title="Reported uses" text={result.uses} />
            <InfoBlock icon={<AlertTriangle className="size-4" />} title="Edibility & safety" text={result.edibility} caution />
            <InfoBlock icon={<ShieldCheck className="size-4" />} title="Safety note" text={result.safety} />

            <p className="js-reveal text-xs leading-relaxed text-forest-foreground/60">{result.warning}</p>

            <Button variant="glow" size="lg" className="js-reveal" onClick={() => selectFile(null)}>
              <RotateCcw /> Try another image
            </Button>
          </div>
        ) : (
          <EmptyResult loading={loading} />
        )}
      </div>
    </div>
  );
}

function InfoBlock({ icon, title, text, caution }: { icon: React.ReactNode; title: string; text: string; caution?: boolean }) {
  return (
    <div
      className={cn(
        "js-reveal rounded-2xl p-4",
        caution ? "border border-caution-border bg-caution text-caution-foreground" : "bg-forest-foreground/5 ring-1 ring-forest-foreground/10",
      )}
    >
      <h5 className="mb-1 flex items-center gap-2 text-sm font-bold">
        {icon} {title}
      </h5>
      <p className={cn("whitespace-pre-line text-sm leading-relaxed", !caution && "text-forest-foreground/80")}>{text}</p>
    </div>
  );
}

function EmptyResult({ loading }: { loading: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useGSAP(
    () => {
      gsap.to(".js-ring", { rotate: 360, duration: loading ? 3 : 18, ease: "none", repeat: -1, transformOrigin: "50% 50%" });
      gsap.to(".js-float", { y: -8, duration: 2.4, ease: "sine.inOut", yoyo: true, repeat: -1 });
    },
    { scope: ref, dependencies: [loading], revertOnUpdate: true },
  );
  return (
    <div ref={ref} className="relative flex min-h-[340px] flex-col items-center justify-center text-center">
      <div className="relative mb-6 grid size-32 place-items-center">
        <svg viewBox="0 0 120 120" className="js-ring absolute inset-0 size-full text-primary-glow/50">
          <circle cx="60" cy="60" r="56" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="2 8" />
          <circle cx="60" cy="4" r="3" fill="currentColor" />
        </svg>
        <Leaf className="js-float size-12 text-primary-glow" strokeWidth={1.25} />
      </div>
      <p className="font-display text-xl">{loading ? "Reading the veins…" : "Awaiting a specimen"}</p>
      <p className="mt-2 max-w-xs text-sm text-forest-foreground/60">
        {loading
          ? "Our model is comparing leaf shape, margin and venation."
          : "Upload a clear photo of a single leaf to see the predicted species, uses and safety notes."}
      </p>
    </div>
  );
}
