import { useRef } from "react";
import { createFileRoute } from "@tanstack/react-router";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { ArrowDown, Camera, Cpu, FileSearch, Leaf, ScanLine, ShieldAlert } from "lucide-react";

import heroImg from "@/assets/leaf-hero.jpg";
import { Button } from "@/components/ui/button";
import { Identifier } from "@/components/leaflens/Identifier";

gsap.registerPlugin(useGSAP, ScrollTrigger);

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "LeafLens AI — Identify a plant from a leaf photo" },
      {
        name: "description",
        content: "Upload a leaf photo to see LeafLens AI's top model predictions and source-linked botanical information.",
      },
      { property: "og:title", content: "LeafLens AI — Identify a plant from a leaf photo" },
      {
        property: "og:description",
        content: "Leaf-image classification with top model predictions and source-linked botanical information.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const STEPS = [
  { icon: Camera, title: "Photograph", text: "Shoot one leaf against a plain background in good, even light." },
  { icon: Cpu, title: "Analyse", text: "An 80-class Vision Transformer analyses the leaf image." },
  { icon: FileSearch, title: "Review", text: "Review the top five model labels and source-linked botanical information." },
];

function Index() {
  const root = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const tl = gsap.timeline({ defaults: { ease: "expo.out" } });
      tl.from(".js-nav", { y: -30, opacity: 0, duration: 0.9 })
        .from(".js-word", { yPercent: 110, duration: 1.1, stagger: 0.07 }, "-=0.5")
        .from(".js-hero-fade", { y: 20, opacity: 0, duration: 0.9, stagger: 0.1 }, "-=0.7")
        .from(
          ".js-hero-img",
          { clipPath: "inset(100% 0% 0% 0% round 28px)", scale: 1.15, duration: 1.4, ease: "expo.inOut" },
          0.2,
        )
        .from(".js-chip", { y: 20, opacity: 0, scale: 0.9, duration: 0.8, stagger: 0.15, ease: "back.out(2)" }, "-=0.5");

      gsap.to(".js-chip-a", { y: -10, duration: 3, yoyo: true, repeat: -1, ease: "sine.inOut" });
      gsap.to(".js-chip-b", { y: 10, duration: 3.6, yoyo: true, repeat: -1, ease: "sine.inOut" });
      gsap.fromTo(".js-hero-scan", { top: "8%" }, { top: "88%", duration: 2.6, yoyo: true, repeat: -1, ease: "sine.inOut" });

      gsap.to(".js-hero-photo", {
        yPercent: 12,
        ease: "none",
        scrollTrigger: { trigger: ".js-hero", start: "top top", end: "bottom top", scrub: true },
      });

      gsap.from(".js-card", {
        y: 60,
        opacity: 0,
        duration: 1,
        stagger: 0.15,
        ease: "power3.out",
        scrollTrigger: { trigger: "#identify", start: "top 80%" },
      });

      gsap.from(".js-step", {
        y: 40,
        opacity: 0,
        duration: 0.9,
        stagger: 0.12,
        ease: "power3.out",
        scrollTrigger: { trigger: ".js-steps", start: "top 85%" },
      });

      gsap.from(".js-notice", {
        scale: 0.96,
        opacity: 0,
        duration: 1,
        ease: "power3.out",
        scrollTrigger: { trigger: ".js-notice", start: "top 90%" },
      });
    },
    { scope: root },
  );

  const headline = [
    ["Know", "the"],
    ["leaf", "in"],
    ["your", "hand."],
  ];

  return (
    <div ref={root} className="paper-grain min-h-screen overflow-x-hidden">
      {/* Nav */}
      <header className="js-nav sticky top-0 z-30 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5">
          <a href="#" className="flex items-center gap-2.5">
            <span className="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground">
              <Leaf className="size-5" />
            </span>
            <span className="font-display text-xl font-semibold">
              LeafLens <span className="italic text-primary">AI</span>
            </span>
          </a>
          <nav className="hidden items-center gap-8 text-sm font-medium text-muted-foreground md:flex">
            <a href="#identify" className="transition hover:text-foreground">Identify</a>
            <a href="#how" className="transition hover:text-foreground">How it works</a>
            <a href="#safety" className="transition hover:text-foreground">Safety</a>
          </nav>
          <Button variant="hero" size="sm" className="h-9 px-4" asChild>
            <a href="#identify">Try it</a>
          </Button>
        </div>
      </header>

      {/* Hero */}
      <section className="js-hero mx-auto grid max-w-6xl items-center gap-12 px-5 pb-20 pt-12 lg:grid-cols-[1.1fr_1fr] lg:pt-20">
        <div>
          <p className="js-hero-fade mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            <span className="size-1.5 rounded-full bg-primary" /> Botanical recognition
          </p>
          <h1 className="text-[clamp(3rem,8vw,6.25rem)] font-light leading-[0.95]">
            {headline.map((line, i) => (
              <span key={i} className="block overflow-hidden pb-[0.12em] -mb-[0.12em]">
                {line.map((w) => (
                  <span
                    key={w}
                    className={`js-word mr-[0.22em] inline-block ${w === "leaf" || w === "hand." ? "italic text-primary" : ""}`}
                  >
                    {w}
                  </span>
                ))}
              </span>
            ))}
          </h1>
          <p className="js-hero-fade mt-7 max-w-md text-lg leading-relaxed text-muted-foreground">
            Upload a leaf photo. LeafLens shows the model's top predictions and a source-linked botanical report.
          </p>
          <div className="js-hero-fade mt-9 flex flex-wrap items-center gap-4">
            <Button variant="hero" size="xl" asChild>
              <a href="#identify">
                <ScanLine /> Identify a leaf
              </a>
            </Button>
            <Button variant="soft" size="xl" asChild>
              <a href="#how">
                How it works <ArrowDown />
              </a>
            </Button>
          </div>
        </div>

        <div className="relative mx-auto w-full max-w-md lg:max-w-none">
          <div className="js-hero-img relative aspect-[4/5] overflow-hidden rounded-[28px] shadow-lift">
            <img
              src={heroImg}
              alt="Backlit green leaf with dewdrops"
              width={1024}
              height={1280}
              className="js-hero-photo absolute inset-0 size-full scale-110 object-cover"
            />
            <div className="js-hero-scan scanline pointer-events-none absolute inset-x-0 h-16 opacity-70 mix-blend-screen" />
            <div className="pointer-events-none absolute inset-6 rounded-2xl border border-forest-foreground/25" />
          </div>

          <div className="js-chip js-chip-a absolute -left-4 top-10 rounded-2xl border border-border bg-card/95 px-4 py-3 shadow-soft backdrop-blur sm:-left-10">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Model</p>
            <p className="font-display text-lg">Vision Transformer</p>
          </div>
          <div className="js-chip js-chip-b absolute -right-3 bottom-12 flex items-center gap-3 rounded-2xl bg-forest px-4 py-3 text-forest-foreground shadow-lift sm:-right-8">
            <span className="grid size-9 place-items-center rounded-full bg-accent text-accent-foreground">
              <ScanLine className="size-4" />
            </span>
            <div>
              <p className="text-xs text-forest-foreground/60">One photo</p>
              <p className="font-semibold">One prediction</p>
            </div>
          </div>
        </div>
      </section>

      {/* Identifier */}
      <section id="identify" className="mx-auto max-w-6xl scroll-mt-24 px-5 pb-24">
        <div className="mb-10 flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <h2 className="text-4xl font-light sm:text-5xl">
            Identify a <span className="italic text-primary">plant</span>
          </h2>
          <p className="max-w-sm text-muted-foreground">JPG, PNG or BMP up to 10 MB. Drag a photo in or browse your files.</p>
        </div>
        <Identifier />
      </section>

      {/* How it works */}
      <section id="how" className="border-y border-border bg-card/60">
        <div className="mx-auto max-w-6xl px-5 py-24">
          <h2 className="mb-14 max-w-xl text-4xl font-light sm:text-5xl">
            Three steps from <span className="italic text-primary">photo</span> to field report.
          </h2>
          <div className="js-steps grid gap-6 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <div key={s.title} className="js-step group rounded-3xl border border-border bg-background p-7 transition-transform duration-500 hover:-translate-y-1.5">
                <div className="mb-10 flex items-center justify-between">
                  <span className="grid size-12 place-items-center rounded-2xl bg-secondary text-primary transition-colors duration-500 group-hover:bg-primary group-hover:text-primary-foreground">
                    <s.icon className="size-5" />
                  </span>
                  <span className="font-display text-5xl font-light italic text-border">0{i + 1}</span>
                </div>
                <h3 className="text-2xl">{s.title}</h3>
                <p className="mt-2 text-muted-foreground">{s.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Safety */}
      <section id="safety" className="mx-auto max-w-6xl scroll-mt-24 px-5 py-20">
        <div className="js-notice flex flex-col gap-5 rounded-3xl border border-caution-border bg-caution p-7 text-caution-foreground sm:flex-row sm:items-center sm:p-9">
          <span className="grid size-14 shrink-0 place-items-center rounded-2xl bg-card text-caution-foreground shadow-soft">
            <ShieldAlert className="size-6" />
          </span>
          <div>
            <h3 className="text-2xl">Never eat a plant based solely on an AI identification.</h3>
            <p className="mt-1 opacity-80">
              Predictions can be wrong. Always confirm a species with a reliable botanical source or a local expert.
            </p>
          </div>
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-5 py-8 text-sm text-muted-foreground sm:flex-row">
          <span className="flex items-center gap-2">
            <Leaf className="size-4 text-primary" /> LeafLens AI
          </span>
          <span>Powered by the Hugging Face ViT classifier</span>
        </div>
      </footer>
    </div>
  );
}
