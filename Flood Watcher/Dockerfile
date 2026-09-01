# ── Flood Watcher frontend ───────────────────────────────────────────────────
#
# Runs the same dev server command you already use locally (`npm run dev`,
# see ../start.sh) — this project has only ever been run this way so far.
#
# NOTE: TanStack Start's production build (`npm run build`) here targets
# Cloudflare Workers by default (see vite.config.ts's nitro comment), which
# a plain Node container like this one can't run as-is. If a real Node
# production build is wanted later, the Nitro preset needs to be switched to
# "node-server" first and the build tested — not done here, since this
# project hasn't used a production build at all up to now.
#
# node:22, not 20 — @tanstack/react-start (and several of its sibling
# packages) declare "engines": { "node": ">=22.12.0" }. node:20 still runs it
# (it's just an EBADENGINE warning, not a hard block), but 22 is what the
# packages themselves say they need, so that's what the image uses.
FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json ./
# `npm ci` demands the lock file match package.json exactly, and the npm
# bundled with node:22-slim is stricter about that check than the npm on the
# Mac this was developed on (10.8.2) — it was rejecting a small, real drift
# (a missing lru-cache entry) that the older local npm quietly let through.
# `npm install` reconciles that drift itself at build time instead of hard
# failing on it. Only affects files inside the image/container — nothing on
# the host Mac is touched.
RUN npm install

COPY . .

EXPOSE 5173

# --host 0.0.0.0 so the dev server is reachable from outside the container,
# not just from inside it.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
