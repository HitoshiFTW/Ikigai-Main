# Deploying Ikigai into the Wild — a from-zero guide

You've done Vast.ai GPU + Jupyter. This is different, and simpler in some ways. Read the
mental model first (5 min), then follow the steps. Every command is copy-paste.

---

## 0. The mental model (how this differs from Vast.ai)

| Vast.ai / Jupyter | This deployment |
|---|---|
| You rent a **GPU** by the hour, open a notebook, run cells, it dies when you stop. | You rent a tiny **CPU** server (a "VPS") that stays on **24/7**, running one program forever. |
| Notebook = interactive. | A **web server** = it sits waiting, and answers HTTP requests from the website. |
| Needs GPU. | **No GPU.** Ikigai runs on 1 CPU core — that's the whole point. |

Two pieces, hosted in two places:

```
   the website (HTML/JS)          the organism (Python + brain)
   ── static, no compute ──       ── a long-running process ──
        Vercel                          a $5 VPS
   ikigai.mura-alife.com  ─── talks over HTTPS ───►  api.ikigai.mura-alife.com
```

**Docker** = a way to freeze the organism + Python + its brain into one "image" (think: a
`.zip` of a whole mini-computer). You build the image once; it then runs *identically*
anywhere. No "works on my machine" — the machine comes inside the box.

- **Image** = the frozen template (built from the `Dockerfile`).
- **Container** = a running copy of an image (the live organism).
- **Volume** = a folder on the host that the container writes to, so data survives restarts.

---

## 1. What you need

1. **A VPS** — 1 CPU / 1GB RAM / 25GB SSD is enough (we measured: 650MB under load).
   Good cheap ones: **Hetzner** (CX22, ~€4/mo — best value), DigitalOcean, Vultr, Linode.
   Pick **Ubuntu 24.04** as the OS.
2. **A Docker Hub account** (free) — `hub.docker.com`. This is where you push the image so
   the VPS can pull it. (Like a model on HuggingFace Hub, but for containers.)
3. **Docker Desktop on your Windows PC** (free) — `docker.com/products/docker-desktop`.
   Install it, launch it once (it runs in the tray). This lets you build the image locally.
4. Your domain **mura-alife.com** (you have it) — to point `api.ikigai.` at the VPS.

---

## 2. Build the image on your PC and push it (Path A — recommended)

Open **PowerShell** in the repo (`c:\neuroseed`). Docker Desktop must be running.

```powershell
# log in to Docker Hub (one time)
docker login

# build the image (this reads the Dockerfile, installs numpy, copies the code + brain).
# replace YOURNAME with your Docker Hub username. the "." means "build from here".
docker build -t YOURNAME/ikigai:latest .

# push it to Docker Hub (~450MB, one-time upload; later pushes are faster)
docker push YOURNAME/ikigai:latest
```

That's it — the organism is now a pullable image. **Test it locally before shipping:**

```powershell
docker run --rm -p 8080:8080 -v ikigai_data:/data YOURNAME/ikigai:latest
```

Then in a browser open `http://localhost:8080/vitals` — you should see JSON stats. Or:

```powershell
# in another PowerShell window: ask it something
curl.exe -X POST http://localhost:8080/ -H "Content-Type: application/json" -d '{\"q\":\"what is the capital of france\"}'
```

You should get `{"answer":"The capital of france is paris.", ...}`. Ctrl-C to stop.

> **What the run flags mean:** `-p 8080:8080` connects your PC's port 8080 to the container's.
> `-v ikigai_data:/data` gives it a persistent folder (the wild life). `--rm` deletes the
> container when it stops (the volume survives).

---

## 3. Set up the VPS

After creating the VPS you get an **IP address** (e.g. `203.0.113.10`) and a root password (or
you added an SSH key). From PowerShell:

```powershell
ssh root@203.0.113.10        # type "yes", then the password
```

You're now on the server. Everything below runs **on the VPS**.

### 3a. Add swap (the 1GB safety net — do this first)

A continual learner slowly uses more RAM. Swap = spillover space on disk so it never crashes.

```bash
fallocate -l 2G /swapfile          # make a 2GB swap file
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab   # make it permanent across reboots
free -h                             # verify: you should see 2.0Gi under "Swap"
```

### 3b. Install Docker

```bash
curl -fsSL https://get.docker.com | sh      # official one-line installer
docker --version                            # verify
```

### 3c. Open the firewall for the port

```bash
apt install -y ufw
ufw allow OpenSSH
ufw allow 8080/tcp
ufw --force enable
```

---

## 4. Run the organism on the VPS

```bash
docker run -d \
  --name ikigai \
  --restart unless-stopped \
  -p 8080:8080 \
  -v ikigai_data:/data \
  YOURNAME/ikigai:latest
```

> `-d` = run in background (detached). `--restart unless-stopped` = if it ever crashes OR the
> RSS watchdog restarts it OR the server reboots, Docker brings it back automatically — and it
> reloads the saved wild life from the volume. This is what makes it *live in the wild* 24/7.

Check it's alive:

```bash
docker logs ikigai              # should show "IKIGAI live in the wild on :8080"
curl http://localhost:8080/vitals
```

---

## 5. Point your domain at it + HTTPS (important)

The website is served over **https**, and browsers **block** an https page from calling a plain
http backend ("mixed content"). So `api.ikigai.mura-alife.com` must be **https** too. Easiest
way — **Cloudflare** (free), which also hides your server's IP:

1. If mura-alife.com isn't already on Cloudflare, add it (free plan) and switch its
   nameservers — Cloudflare walks you through it.
2. In Cloudflare DNS, add a record:
   - Type **A**, Name **api.ikigai**, IPv4 = your VPS IP, Proxy status **Proxied** (orange cloud).
3. SSL/TLS mode: set to **Flexible** (Cloudflare does https to the browser, talks http to your
   origin on 8080 — 8080 is a Cloudflare-supported port).

Now `https://api.ikigai.mura-alife.com/vitals` works with a padlock. Put that URL in the
website's `IKIGAI_API` constant.

> **No Cloudflare?** Alternative: run a **Caddy** reverse-proxy container in front — it fetches a
> free Let's Encrypt certificate automatically. Ask me and I'll write the 6-line Caddy setup.

---

## 6. Verify end-to-end

```bash
# from anywhere:
curl -X POST https://api.ikigai.mura-alife.com/ \
  -H "Content-Type: application/json" \
  -d '{"q":"the capital of qualan is mendaro"}'
# -> {"answer":"Got it -- the capital of qualan is mendaro","chose":"learn",...}

curl -X POST https://api.ikigai.mura-alife.com/ \
  -H "Content-Type: application/json" \
  -d '{"q":"what is the capital of qualan"}'
# -> {"answer":"The capital of qualan is mendaro.","chose":"answer",...}
```

If those two work, it's **live in the wild** — learning from strangers, over the internet.

---

## 7. Day-2 operations (the few commands you'll actually use)

```bash
docker logs -f ikigai            # watch it live (Ctrl-C to stop watching, not the container)
docker stats ikigai              # live CPU + RAM (watch RAM stay under ~850MB)
docker restart ikigai            # restart it
docker stop ikigai               # stop it
docker exec -it ikigai bash      # open a shell INSIDE the container (poke around)

# to ship an update (after you rebuild + push a new image from your PC):
docker pull YOURNAME/ikigai:latest
docker stop ikigai && docker rm ikigai
# ...then re-run the `docker run -d ...` command from step 4. The volume (wild life) is untouched.
```

**Back up the wild life** (what it learned from everyone) anytime:

```bash
docker run --rm -v ikigai_data:/data -v $(pwd):/backup busybox \
  cp /data/wild_organism.ikg /backup/wild_backup.ikg
```

---

## Path B — build on the VPS instead (if you skip Docker Desktop)

Get the needed files onto the VPS (from your PC, in PowerShell), then build there:

```powershell
# copy only what the image needs (not the whole repo)
scp integrate.py root@203.0.113.10:/root/ikigai/
scp organism.ikg root@203.0.113.10:/root/ikigai/
scp Dockerfile .dockerignore root@203.0.113.10:/root/ikigai/
scp -r ikigai root@203.0.113.10:/root/ikigai/ikigai
scp -r experiments/wild root@203.0.113.10:/root/ikigai/experiments/wild
```

Then on the VPS:

```bash
cd /root/ikigai
docker build -t ikigai:latest .
# then the same `docker run -d ...` from step 4, using image name  ikigai:latest
```

> Building on a 1GB box is fine (numpy installs from a prebuilt wheel, no compiling). It just
> uses some RAM during `pip install` — the swap from 3a covers it.

---

## Recap

1. Build image on PC → push to Docker Hub (step 2).
2. VPS: add swap, install Docker, open port 8080 (step 3).
3. `docker run -d --restart unless-stopped -v ikigai_data:/data ...` (step 4).
4. Cloudflare: `api.ikigai` → VPS IP, proxied, Flexible SSL (step 5).
5. Put `https://api.ikigai.mura-alife.com` in the website. Done.

The organism now lives on a $5 CPU box, answers or abstains (never bluffs), and learns from
every stranger — forever. That price tag *is* the pitch.
