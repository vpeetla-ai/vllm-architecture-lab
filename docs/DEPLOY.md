# Deploy — vLLM Architecture Lab

## Live URLs

| Service | URL |
|---------|-----|
| **Demo UI** | https://vllm-architecture-lab.vercel.app |
| **API** | https://vllm-architecture-lab-api.onrender.com |

## Vercel (demo)

```bash
cd demo
npx vercel --prod --yes
```

Project name: `vllm-architecture-lab` (do **not** deploy to the shared `demo` project).

## Render (API)

1. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect `vpeetla-ai/vllm-architecture-lab` — `render.yaml` auto-detected
3. Wait for deploy; verify: `curl https://vllm-architecture-lab-api.onrender.com/health`

## Wire live sim

Update `demo/config.js`:

```js
const VLLM_LAB_API = "https://vllm-architecture-lab-api.onrender.com";
```

Redeploy Vercel after config change.
