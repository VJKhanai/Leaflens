<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Architecture
- Leaf predictions come from an external model server called via `src/lib/leaflens.ts` (`VITE_PREDICT_URL`, default `/predict`); keep all request/validation logic there so UI stays presentation-only.
- Animations use GSAP via `useGSAP` scoped to a component ref, so they clean up on unmount and stay SSR-safe.
