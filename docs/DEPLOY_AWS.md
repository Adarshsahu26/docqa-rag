# Deploying DocQA on AWS (EC2 + Docker Compose)

> **Status.** These steps were written carefully from AWS and Docker's standard workflow but **have not been run**:
> there was no AWS account to try them on. Every command is a common one, and the [troubleshooting table](#12-troubleshooting)
> covers the usual problems. Remove this note once you have been through the guide yourself.

**What you end up with:** one EC2 instance running the Compose stack from this repository, answering at
`http://<public-ip>:8000`, protected by an API key, with the search index on a Docker volume that survives reboots
and updates.

**What you need:** an AWS account, about an hour for the first run, and (optionally) an Anthropic or OpenAI-compatible
API key. Without a key the service still runs, in extractive mode. A small always-on instance plus its disk costs a
few dollars a month. Check the current EC2 and EBS prices, and read step 11 so nothing keeps billing after you finish.

## 1. Set a billing alert first

AWS console, then **Billing and Cost Management**, then **Budgets**, then **Create budget**. Use the **Zero spend budget**
template (it emails you as soon as anything is charged) or a **Monthly cost budget** with a small limit. Do this
before creating anything else.

## 2. Launch the instance

EC2, then **Launch instance**, in a region close to you:

| Setting | Value |
|---|---|
| Name | `docqa` |
| Image (AMI) | **Ubuntu Server**, current LTS, 64-bit (x86). The default login user is `ubuntu`. Amazon Linux uses `ec2-user` and `dnf`, so these steps would differ |
| Instance type | **t3.small** (2 GiB RAM). Use an x86 type, not an ARM (`t4g`) one. A `t3.micro` (1 GiB) is tight; if you use one, add swap (see the troubleshooting table) |
| Key pair | **Create new key pair**, name `docqa-key`, type RSA, format `.pem`. The file downloads once: keep it safe |
| Network settings | Click **Edit**. Make sure *Auto-assign public IP* is enabled. Create a new security group with two inbound rules: **SSH (22)** and **Custom TCP 8000**, both with source **My IP** |
| Storage | **20 GiB** gp3. The image with PyTorch and the model is a few GB, and Docker's build cache needs room |

Click **Launch instance** and wait for the state *Running*. Copy the **Public IPv4 address**.

Allowing only **My IP** keeps strangers out while you test. If your home IP changes (most home connections do), edit
the rules in EC2, then **Security Groups**, then **Inbound rules**. The public address also changes whenever the instance is
stopped and started, unless you attach an Elastic IP.

## 3. Connect

macOS or Linux:

```bash
chmod 400 docqa-key.pem
ssh -i docqa-key.pem ubuntu@<public-ip>
```

Windows 10/11 (PowerShell has an SSH client built in):

```powershell
ssh -i .\docqa-key.pem ubuntu@<public-ip>
```

If Windows says the key file permissions are too open, run
`icacls .\docqa-key.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"` and try again.

The browser-based **EC2 Instance Connect** also works, but it connects from AWS's own address range, so a
"My IP"-only SSH rule blocks it. Using your own SSH client is simpler.

## 4. Install Docker

On the server:

```bash
sudo apt-get update
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit
```

Connect again (step 3) so the new group membership applies, then check:

```bash
docker --version
docker compose version
```

## 5. Get the code onto the server

If the project is on GitHub:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

For a private repository, or if you only have the zip file, copy the zip from your computer instead. Run the `scp`
line on your own computer, then the rest on the server:

```bash
scp -i docqa-key.pem docqa-rag_snapshot-3.zip ubuntu@<public-ip>:~
# on the server:
sudo apt-get install -y unzip
unzip docqa-rag_snapshot-3.zip
cd docqa-rag
```

## 6. Configure

```bash
cp .env.example .env
nano .env
```

Remove the leading `#` from the lines you need and fill them in. No quotes, no spaces around `=`:

```
DOCQA_API_KEY=<paste the output of:  openssl rand -hex 24>
ANTHROPIC_API_KEY=<your Anthropic key>
DOCQA_LLM_PROVIDER=anthropic
DOCQA_LLM_MODEL=claude-haiku-5-5
```

For an OpenAI-compatible provider use `OPENAI_API_KEY`, `DOCQA_LLM_PROVIDER=openai`, `DOCQA_LLM_MODEL` (and
`OPENAI_BASE_URL` for anything that is not OpenAI) instead. For no LLM at all, leave those lines out.

Always set `DOCQA_API_KEY`. Without it, anyone who finds the address can upload files and spend your LLM credits.
Do not set `DOCQA_DATA_DIR` here: the image already stores the index in the `/data` volume. Then lock the file down:

```bash
chmod 600 .env
```

`.env` is in `.gitignore` and `.dockerignore`, so it is never committed and never baked into the image.

## 7. Start it

```bash
docker compose up -d --build
docker compose logs -f
```

The first build takes several minutes (it downloads the CPU build of PyTorch and the embedding model). Wait for
`Pipeline ready: embedder=sentence-transformers/all-MiniLM-L6-v2 ...` and `Uvicorn running on http://0.0.0.0:8000`,
then press Ctrl+C to stop following the log. The container keeps running.

## 8. Try it

On the server:

```bash
curl http://localhost:8000/health
```

It should look like `{"status":"ok","embedder":"sentence-transformers/all-MiniLM-L6-v2","llm":"anthropic:claude-haiku-5-5","documents":0,"chunks":0}`.
`docker compose ps` shows the container as *healthy* after about a minute.

Upload a sample document and ask about it (still on the server):

```bash
KEY=$(grep ^DOCQA_API_KEY= .env | cut -d= -f2)
curl -H "X-API-Key: $KEY" -F "file=@sample_docs/pump_station_alarm_codes.txt" http://localhost:8000/documents
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"question": "What does alarm code P-101 mean?"}' http://localhost:8000/ask
```

From your own computer, open `http://<public-ip>:8000/docs`, click **Authorize**, paste the key, and use the
endpoints from the browser. If `/health` works on the server but not from your computer, the security group is
the cause (see the table).

## 9. Day-to-day

```bash
docker compose logs -f docqa     # follow the log
docker compose ps                # status and health
docker compose restart           # restart the service
```

**Reboots.** The compose file uses `restart: unless-stopped` and Docker starts at boot, so the service comes back by
itself. Test it once with `sudo reboot`, reconnect after a minute, and check `/health`: the document count should be
the same as before.

**Updating** after you change the code:

```bash
git pull                          # or upload a new zip and unzip it over the old folder
docker compose up -d --build
```

The index is on a volume, so it survives the rebuild.

**Backing up the index.** Find the volume's name (Compose prefixes it with the folder name, for example
`docqa-rag_docqa-data`), stop the service so no save is half-done, archive, and start again:

```bash
docker volume ls
docker compose stop
docker run --rm -v docqa-rag_docqa-data:/data -v "$PWD":/backup alpine tar czf /backup/docqa-data.tgz -C /data .
docker compose start
```

Restoring is the reverse: stop the service, then
`docker run --rm -v docqa-rag_docqa-data:/data -v "$PWD":/backup alpine sh -c "rm -rf /data/* && tar xzf /backup/docqa-data.tgz -C /data"`,
then start it. Copy `docqa-data.tgz` off the server (with `scp`) if you care about it.

## 10. HTTPS (recommended before real use; optional and untested)

Plain HTTP sends your API key and documents unencrypted. For anything beyond a quick test, put a reverse proxy with
a certificate in front. Caddy does this with very little configuration. You need a domain name whose DNS `A` record
points at the instance (attach an **Elastic IP** first so the address stops changing).

1. In the security group, allow **HTTP (80)** and **HTTPS (443)** from anywhere (the certificate authority must be
   able to reach the server), and remove the port 8000 rule.
2. Create `Caddyfile` in the project folder (this also caps the request size, which the app alone cannot fully do):

   ```
   docqa.example.com {
       request_body {
           max_size 12MB
       }
       reverse_proxy docqa:8000
   }
   ```

3. Create `docker-compose.caddy.yml` next to `docker-compose.yml`:

   ```yaml
   services:
     caddy:
       image: caddy:2
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./Caddyfile:/etc/caddy/Caddyfile:ro
         - caddy-data:/data
       depends_on:
         - docqa
       restart: unless-stopped

   volumes:
     caddy-data:
   ```

4. Start both files together: `docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d`.

Then use `https://docqa.example.com/docs`. Replace the domain with yours.

## 11. Cost control and clean-up

- **Stopping** the instance (EC2, **Instance state**, **Stop**) ends the compute charge, but the disk (EBS) keeps billing
  and the public IP changes on the next start.
- **Terminating** the instance deletes it and, by default, its root disk. Do this when you are done with the
  project, after taking a backup if you want the index.
- **Release the Elastic IP** if you created one, and delete leftover volumes and snapshots. AWS charges a small
  hourly fee for public IPv4 addresses, including idle ones.
- Look at **Billing** a day after you finish, and keep the Budgets alert from step 1 on.

## 12. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `Permission denied (publickey)` | Wrong user (it is `ubuntu`), wrong key file, or key permissions (`chmod 400`; on Windows the `icacls` line in step 3) |
| SSH or `curl` to the public IP hangs | The security group has no rule for you, or your IP changed (edit the rule), or the instance was restarted and has a new public IP |
| `permission denied while trying to connect to the Docker daemon` | You did not reconnect after `usermod`. Log out and in, or prefix the command with `sudo` |
| `env file .env not found` | Run `cp .env.example .env` in the project folder |
| The container exits at start | `docker compose logs docqa`. A common line is `DOCQA_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set`: fix `.env`, then `docker compose up -d` |
| The build is killed or the instance freezes (out of memory) | Use a larger type, or add swap: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| `no space left on device` | `docker system prune -af` removes unused images and build cache, or grow the EBS volume (EC2, **Volumes**, **Modify volume**, then `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1` on a typical Nitro instance) |
| `401 Missing or invalid API key` | Send the `X-API-Key` header with the value from `.env` (or use **Authorize** in `/docs`) |
| `502` from `/ask` | The model call failed. `docker compose logs docqa` shows why: wrong key, wrong model name, no credit, or a rate limit |
| `/ask` says "I couldn't find that" for something you know is in the documents | Call `/search` with the same text and look at the scores. If the right chunk scores just under the cut-off, lower `DOCQA_MIN_SCORE` in `.env` |
| `413` on upload | The file is over `DOCQA_MAX_UPLOAD_MB` (default 10) |

## 13. Other options

The same Docker image runs on container services. Whichever you choose, keep these three rules: **one instance
only** (the index is held in memory), a **persistent volume mounted at `/data`** (EFS on ECS, Azure Files on Azure
Container Apps), and **`/health` as the health-check path**. Pass the keys as secrets (AWS Secrets Manager, Azure Key
Vault) rather than baking them into anything. Check each service's current documentation for the exact settings.

## 14. What you can truthfully say afterwards

Once the service has been running on EC2 and answering questions, "deployed a containerised FastAPI RAG service on
AWS EC2 with Docker Compose" is accurate. Before that, it is not. Keep a screenshot of `/docs` and of a grounded
answer from your own deployment, and stop or terminate the instance when you no longer need it so that the link in
a CV does not point at a bill. The README of your repository can say "deployment guide: docs/DEPLOY_AWS.md" in the
meantime.
