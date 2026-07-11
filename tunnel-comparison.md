# Exposing a Web Terminal to iPhone: Free Tunneling Methods Compared

**Scenario:** A Linux localhost runs an xterm.js-based web terminal on port 8080 (HTTP/WebSocket). The goal is to expose it so that an iPhone running Safari or Chrome can access it reliably, with persistent/always-on availability as the primary use case.

This document compares seven free methods: **Cloudflare Tunnel (cloudflared)**, **ngrok**, **Tailscale Funnel**, **ZeroTier**, **frp**, **Serveo**, and **DIY SSH reverse tunneling**. Each is evaluated on its architecture, free-tier specifics, setup complexity, iOS compatibility, and persistent-vs-ephemeral viability.

---

## 1. Cloudflare Tunnel (cloudflared)

### How It Works

`cloudflared` establishes outbound, post-quantum encrypted connections from your Linux host to Cloudflare's global edge network. No inbound ports need to be opened, and no public IP is required. Traffic from the internet enters at any Cloudflare edge data center and is routed through the tunnel to your local service. Cloudflare automatically applies CDN caching, WAF, DDoS protection, and automatic TLS certificates.

There are two modes:
- **Quick Tunnel (TryCloudflare):** Generates a random `trycloudflare.com` subdomain for immediate testing without any account.
- **Production Tunnel:** Requires a Cloudflare account and a domain hosted on Cloudflare DNS. Maps a public hostname to your local service via the Cloudflare dashboard or CLI.

### Free-Tier Specifics

| Aspect | Quick Tunnel | Production Tunnel (Free Plan) |
|---|---|---|
| **Bandwidth** | No hard cap | **Unlimited** — no bandwidth limits on free tier |
| **Session Duration** | Persistent while `cloudflared` runs | Persistent — always-on by design |
| **Concurrent Requests** | Hard limit of 200 in-flight HTTP requests | No documented limit |
| **SSE Support** | ❌ Not supported | ✅ Supported |
| **WebSocket Support** | ✅ Supported (production only; quick tunnel has no explicit restriction but the 200-request cap may be problematic for long-lived WS connections) | ✅ Supported without restriction |
| **Custom Domain** | ❌ Random `trycloudflare.com` subdomain only | Requires a domain on Cloudflare DNS (a free Cloudflare account is sufficient; you can use any existing domain or register one) |
| **TLS/HTTPS** | Automatic | Automatic |

Cloudflare's own documentation states: *"There are no bandwidth limits"* for production tunnels, and the 200-concurrent-request limit applies only to Quick Tunnels (not production tunnels). A Cloudflare Team member confirmed this in the community forums.

### Linux Setup (Production Tunnel)

```bash
# Install cloudflared
sudo unzip cloudflared-linux-amd64.deb -d /usr/local/bin/  # Debian/Ubuntu
# or: sudo rpm -i cloudflared.rpm                          # RHEL/Fedora

# Authenticate and create tunnel (interactive — prints a token)
cloudflared tunnel login

# Create the tunnel
cloudflared tunnel create web-terminal

# Configure routing (create /etc/cloudflared/config.yml):
# service:
#   - hostname: terminal.example.com        # your Cloudflare-hosted domain
#     url: http://localhost:8080

# Run as a persistent systemd service
sudo cloudflared tunnel run web-terminal
```

**Quick Tunnel (no account needed, ephemeral):**
```bash
cloudflared tunnel --url http://localhost:8080
# Output: https://random-name.trycloudflare.com
```

### iOS Access Method

- **Browser URL:** The production tunnel provides a public HTTPS URL (`https://terminal.example.com`). Open directly in Safari or Chrome — no app required.
- **WebSocket Support:** xterm.js WebSockets work natively through Cloudflare's edge; no special configuration needed.

### Persistent vs. Ephemeral

**Excellent for persistent/always-on use.** Production tunnels are designed to run indefinitely as a systemd service. Each tunnel maintains four long-lived connections to two Cloudflare data centers for built-in redundancy. The `cloudflared` process auto-reconnects on network failures. Combined with unlimited bandwidth, this is the strongest candidate for always-on deployment among cloud-based tools.

---

## 2. ngrok

### How It Works

ngrok runs a daemon (`ngrok`) on your Linux host that creates an outbound-encrypted tunnel to ngrok's cloud infrastructure. When someone visits the ngrok-assigned URL, traffic is routed through ngrok's servers to your local service. ngrok handles TLS termination at the edge and forwards decrypted traffic to your origin.

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Bandwidth** | **1 GB/month** data transfer out |
| **Endpoints** | Up to 3 online endpoints |
| **HTTP Requests** | 20,000 requests/month |
| **TCP Connections** | 5,000 connections/month |
| **TLS Connections** | ❌ Not available on free tier |
| **Custom Domain** | ❌ Only auto-assigned `*.ngrok-free.app` dev domain |
| **Interstitial Page** | ⚠️ Displays a warning page for all HTML browser traffic (can be bypassed with `ngrok-skip-browser-warning: 1` header) |
| **Session Duration** | Persistent while ngrok runs; no explicit session timeout |
| **Rate Limits** | 4,000 HTTP requests/min, 100 TCP connections/min |

The 1 GB/month transfer cap and 5,000 TCP connection/month limit are significant constraints for a web terminal that maintains long-lived WebSocket connections. xterm.js sessions can easily consume bandwidth through terminal output streaming, file transfers, and session persistence.

### Linux Setup

```bash
# Download and install (replace architecture as needed)
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
sudo cp ngrok /usr/local/bin/

# Authenticate (requires signup at https://ngrok.com)
ngrok config add-authtoken YOUR_TOKEN

# Expose port 8080
ngrok http 8080
# Output: https://your-assigned-name.ngrok-free.app -> localhost:8080
```

**As a persistent systemd service:**
```bash
sudo tee /etc/systemd/system/ngrok.service <<EOF
[Unit]
Description=ngrok Tunnel
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ngrok http 8080 --config ~/.config/ngrok/ngrok.yml
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now ngrok.service
```

### iOS Access Method

- **Browser URL:** The free plan provides a public HTTPS URL (`https://your-assigned-name.ngrok-free.app`). Open in Safari or Chrome.
- ⚠️ **Interstitial Warning:** Safari/Chrome will show an interstitial warning page before reaching the terminal. For WebSocket-based xterm.js (which sends non-HTML requests), this can be bypassed by adding the `ngrok-skip-browser-warning: 1` header — but this requires configuring the iPhone browser or a proxy, which is impractical for casual access.

### Persistent vs. Ephemeral

**Moderate for persistent use.** The tunnel stays active while ngrok runs, and auto-reconnect works. However, the **1 GB/month transfer cap** and **5,000 TCP connection/month limit** make it risky for always-on terminal use. A single long SSH session with file transfers could exhaust these limits quickly. One-off or light-use scenarios are viable; production-grade persistent access is not.

---

## 3. Tailscale Funnel

### How It Works

Tailscale Funnel is an overlay-network feature that routes traffic from the broader internet to a local service on your Tailscale network (tailnet). Unlike traditional reverse proxies, it uses a TCP proxy and Funnel relay servers. When someone accesses the Funnel URL, their device connects to a public Funnel relay server, which then establishes an encrypted TCP proxy tunnel back to your device over the Tailscale network.

**Important architectural distinction:** Tailscale Funnel is **not a reverse proxy** — it's a mesh-network exposure mechanism. The relay server does not decrypt traffic; it maintains end-to-end encryption between the client and your device. Traffic flows: `Internet → Funnel Relay Server → Encrypted TCP Proxy → Your Device → Local Service`.

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Cost** | ✅ Available on Tailscale Personal (free) plan — up to 100 devices, 3 users |
| **Bandwidth** | ⚠️ "Subject to non-configurable bandwidth limits" (no published Mbps cap; Tailscale does not publish specific throughput numbers for Funnel) |
| **Session Duration** | Persistent while Funnel is enabled |
| **HTTPS Required** | ✅ Mandatory — only TLS-encrypted connections allowed |
| **Allowed Ports** | Only ports **443**, **8443**, and **10000** |
| **Domain Format** | Must use DNS names in your tailnet domain (`your-tailnet.ts.net`) |
| **Beta Status** | ⚠️ Still in beta (as of 2026) — subject to change |
| **Let's Encrypt Rate Limits** | Repeated certificate requests may trigger Let's Encrypt rate limits (~34-hour cooldown) |

### Linux Setup

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Enable Funnel for your tailnet (interactive approval)
sudo tailscale funnel

# Expose port 8080 via Funnel on port 443
sudo tailscale funnels --port=443:localhost:8080

# Or using the newer CLI syntax:
sudo tailscale funnel http://localhost:8080
```

The generated Funnel URL will be in the format `https://your-service.your-tailnet.ts.net`.

### iOS Access Method

- **Browser URL:** Safari or Chrome can access the Funnel URL (`https://*.ts.net`). No Tailscale app required on the iPhone — Funnel is publicly accessible.
- ⚠️ **WebSocket Consideration:** Since Funnel only allows TLS connections and operates over TCP proxy, WebSockets should work but may experience higher latency due to the relay hop. The beta status means WebSocket handling could change.

### Persistent vs. Ephemeral

**Good for persistent use**, but with caveats. Funnel stays active as long as Tailscale runs, and it's designed for always-on sharing. However, the **beta status**, **non-configurable bandwidth limits**, and **port restrictions** (only 443/8443/10000) make it less predictable than cloudflared. If your xterm.js app requires a specific port other than these three, you'd need to reconfigure it or use a local reverse proxy.

---

## 4. ZeroTier

### How It Works

ZeroTier is a **virtual Ethernet (Layer 2) overlay network**, not a reverse proxy or tunneling service in the traditional sense. It creates a private virtual LAN across devices that join the same ZeroTier network. Each device gets a virtual IP address, and services on your Linux host become reachable at that virtual IP from any other device on the network.

**Critical architectural distinction:** ZeroTier does **not** expose services to the public internet. It creates a private mesh network — both the Linux host and the iPhone must be members of the same ZeroTier network. There is no public URL; access is via the assigned ZeroTier IP address (e.g., `10.x.x.x`).

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Networks** | 1 network |
| **Devices** | 10 devices per network |
| **Admin Seats** | 1 |
| **Bandwidth** | Not metered (but relay traffic may be capacity-limited if direct P2P fails) |
| **Traffic Type** | Virtual Ethernet (Layer 2) — broadcasts/multicast supported |

ZeroTier's free tier was reduced in 2025–2026 from 25 devices/unlimited networks to 10 devices/1 network. Existing users may be grandfathered under the old limits.

### Linux Setup

```bash
# Install ZeroTier
curl -s https://install.zerotier.com | sudo bash

# Join your network (replace with your network ID from https://central.zerotier.com)
sudo zerotier-cli join YOUR_NETWORK_ID

# Authorize the device in ZeroTier Central web console
# Then access your terminal at: http://<zerotier-assigned-ip>:8080
```

### iOS Access Method

- **Requires Native App:** The iPhone must have the **ZeroTier app** installed from the App Store, join the same network, and be authorized in the ZeroTier Central console.
- **Access URL:** `http://<zerotier-assigned-ip>:8080` — no public URL is generated.
- ⚠️ **Not browser-accessible without the app.** Safari or Chrome cannot reach the terminal unless the device is first connected via the ZeroTier network through the native app.

### Persistent vs. Ephemeral

**Moderate for persistent use.** ZeroTier maintains persistent overlay connections, and the virtual network stays up as long as both peers are online and authorized. However, because **both devices must be on the same private network**, this is fundamentally different from the tunneling/proxy approaches. It's excellent for trusted-device scenarios (your own iPhone) but unsuitable if you want to share access with others who don't join your network.

---

## 5. frp (Fast Reverse Proxy)

### How It Works

frp is an open-source, high-performance reverse proxy written in Go. Unlike the cloud-hosted tools above, **frp requires you to self-host both components**: a server (`frps`) on a machine with a public IP (typically a VPS), and a client (`frpc`) on your Linux localhost. The client connects outbound to the server, which then listens on a port or domain and forwards incoming traffic through the established connection back to your local service.

**Infrastructure note:** While frp software is free and open-source (MIT licensed), it requires a VPS with a public IP address (~$3–5/month). This infrastructure cost must be factored into the "free" assessment.

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Software Cost** | Free and open-source (MIT) |
| **Infrastructure** | Requires a VPS (~$3–5/month) |
| **Bandwidth** | Unlimited — depends on your VPS plan |
| **Session Duration** | Persistent while frpc runs |
| **Concurrent Connections** | No hard limit (depends on VPS resources) |
| **Protocols Supported** | TCP, UDP, HTTP, HTTPS, WebSockets |

The "free" assessment here is nuanced: the software has no limitations, but you must bear infrastructure costs. For someone who already has a VPS, frp is effectively free.

### Linux Setup (Client Side)

```bash
# Download frp (replace version/architecture as needed)
wget https://github.com/fatedier/frp/releases/download/v0.65.0/frp_0.65.0_linux_amd64.tar.gz
tar -xzf frp_0.65.0_linux_amd64.tar.gz
cd frp_0.65.0_linux_amd64/

# Create client configuration (frpc.toml):
cat > frpc.toml <<EOF
serverAddr = "your-vps-ip"
serverPort = 7000
auth.method = "token"
auth.token = "your-secret-token"

[[proxies]]
name = "web-terminal"
type = "tcp"
localPort = 8080
remotePort = 8080
EOF

# Start the client
./frpc -c frpc.toml
```

**Server-side (`frps.toml` on VPS):**
```toml
bindPort = 7000
webServer.addr = "0.0.0.0"
webServer.port = 7500
auth.method = "token"
auth.token = "your-secret-token"
```

### iOS Access Method

- **Browser URL:** The VPS's public IP (or domain) with the mapped remote port: `http://<vps-ip>:8080`. For HTTPS, you'd need to add a reverse proxy (e.g., nginx/Caddy) on the VPS in front of frp.
- ⚠️ **No automatic TLS.** Unlike cloudflared or ngrok, frp does not provide HTTPS out of the box — you must configure it yourself on the VPS.

### Persistent vs. Ephemeral

**Excellent for persistent use**, assuming you maintain your VPS. frp is designed for always-on deployment with connection multiplexing and health checks. Combined with systemd service management on both client and server, this is a robust production-grade solution. The tradeoff is the ongoing infrastructure cost and operational overhead of managing your own VPS.

---

## 6. Serveo

### How It Works

Serveo is a public SSH reverse port-forwarding bridge. You connect to `serveo.net` via SSH with a `-R` flag, and Serveo instantly assigns you a public subdomain that routes traffic through the SSH tunnel to your local service. No registration or software download is required — any OpenSSH client works.

Serveo also supports WireGuard as an alternative transport layer (requires account signup), which can provide more efficient persistent tunnels than raw SSH.

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Active Tunnels** | 3 simultaneous tunnels |
| **Custom Subdomains** | ✅ Available (deterministic based on IP/username) |
| **Bandwidth** | Not explicitly limited (but reliability concerns apply) |
| **Session Duration** | Persistent while SSH connection is alive |
| **Interstitial Page** | ⚠️ Warning page for anonymous tunnels (bypassable with `X-Serveo-Warning: false` header or registered account) |
| **Registration** | Not required for basic use |
| **Reliability** | ⚠️ Known intermittent outages and reliability issues since 2019; community reports of inconsistent availability |

Serveo has a Pro plan ($60/year or $6 one-time "10-day pass") that removes interstitial warnings, increases tunnel limits to 10, and provides priority support. However, the free tier's documented reliability concerns are significant for always-on use.

### Linux Setup

```bash
# Basic one-liner (no registration needed):
ssh -R 80:localhost:8080 serveo.net

# Output includes your public URL, e.g.:
# Forwarding for http://your-subdomain.serveo.net to localhost:8080

# For persistent tunnels, use autossh:
autossh -M 0 \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -R 80:localhost:8080 \
    serveo.net
```

### iOS Access Method

- **Browser URL:** Serveo provides a public HTTPS URL (`https://your-subdomain.serveo.net`). Open directly in Safari or Chrome.
- ⚠️ **Interstitial Warning:** Anonymous tunnels show a browser warning page. Registered users or those using the Pro plan avoid this.

### Persistent vs. Ephemeral

**Poor for persistent/always-on use.** While Serveo *can* maintain persistent tunnels via autossh, its documented reliability issues — including intermittent outages and inconsistent service availability since 2019 — make it unsuitable as a reliable always-on solution. It's better suited for one-off demos, quick testing, or scenarios where occasional downtime is acceptable. The SSH-based architecture also means that if the tunnel drops, you need autossh to reconnect (which adds complexity).

---

## 6b. LocalXpose

### How It Works

LocalXpose is a cloud-hosted tunneling service similar to ngrok and cloudflared. A daemon (`loclx`) runs on your Linux host, creating an outbound encrypted tunnel to LocalXpose's infrastructure. Traffic from the internet enters at the LocalXpose edge and is forwarded to your local service via a public URL. Unlike SSH-based approaches, LocalXpose provides a purpose-built web dashboard for traffic inspection, request replay, and tunnel management.

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Active Tunnels** | 2 simultaneous HTTP tunnels |
| **Bandwidth** | ✅ Unlimited (no data cap on free tier) |
| **Session Duration** | Persistent while `loclx` runs |
| **Custom Subdomains** | ❌ Only auto-assigned `*.loclx.io` subdomains (custom subdomains require Pro plan) |
| **TLS/HTTPS** | ✅ Automatic SSL encryption |
| **Interstitial Page** | ❌ No interstitial page on free tier |
| **Protocols Supported** | HTTP, HTTPS (TCP and UDP tunnels require Pro plan) |
| **Traffic Inspection** | Real-time web dashboard included |

LocalXpose's free tier is notable for offering unlimited bandwidth with only a 2-tunnel limit. This is more generous than ngrok's 1 GB/month cap but less flexible than cloudflared (which supports TCP and WebSocket tunnels on the free tier). The lack of custom subdomains on the free plan means your URL changes unless you use the auto-assigned one consistently.

### Linux Setup

```bash
# Install via npm or download binary
npm install -g loclx
# or: download from https://localxpose.io/download

# Authenticate (requires signup at https://localxpose.io/signup)
loclx account login

# Expose port 8080 as an HTTP tunnel
loclx tunnel http --to localhost:8080
# Output: https://random-subdomain.loclx.io -> localhost:8080
```

**As a persistent systemd service:**
```bash
sudo tee /etc/systemd/system/loclx.service <<EOF
[Unit]
Description=LocalXpose Tunnel
After=network-online.target

[Service]
ExecStart=/usr/local/bin/loclx tunnel http --to localhost:8080
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now loclx.service
```

### iOS Access Method

- **Browser URL:** LocalXpose provides a public HTTPS URL (`https://random-subdomain.loclx.io`). Open directly in Safari or Chrome — no app required.
- ⚠️ **WebSocket support** is not explicitly documented for the free tier. The free plan supports HTTP/HTTPS tunnels only (TCP and UDP require Pro). Since xterm.js uses WebSockets over an HTTP upgrade, basic WebSocket support may work through the HTTP tunnel, but this is unverified on the free tier.

### Persistent vs. Ephemeral

**Good for persistent use.** LocalXpose is designed for always-on deployment with a background daemon that stays active 24/7. The unlimited bandwidth on the free tier is a significant advantage over ngrok. However, the 2-tunnel limit and HTTP-only restriction (no raw TCP/UDP on free) may be constraining for advanced terminal use cases. The unverified WebSocket support on the free tier is a potential risk — if WebSockets don't work reliably through HTTP tunnels, this method would need a Pro plan upgrade.

---

## 7. DIY SSH Reverse Tunneling

### How It Works

DIY SSH reverse tunneling uses OpenSSH's built-in remote port forwarding (`-R`) to expose a local service through an intermediate VPS with a public IP. Your Linux host initiates an outbound SSH connection to the VPS, carrying a reverse tunnel that binds a port on the VPS and forwards traffic back to `localhost:8080`. The iPhone connects to `<vps-ip>:<remote-port>`, and traffic flows through the SSH tunnel.

This is the most manual approach but offers maximum control and transparency. It requires no third-party tunneling software beyond OpenSSH (which is preinstalled on virtually all Linux distributions).

**Architecture:**
```
iPhone/Safari → VPS:7000 → [SSH reverse tunnel] → localhost:8080 (Linux host)
```

### Free-Tier Specifics

| Aspect | Limit |
|---|---|
| **Software Cost** | Free — uses OpenSSH (preinstalled on Linux) |
| **Infrastructure** | Requires a VPS (~$3–5/month) |
| **Bandwidth** | Unlimited — depends on VPS plan and SSH connection |
| **Session Duration** | Persistent while SSH connection is alive |
| **Concurrent Connections** | Limited only by VPS resources and SSH configuration |
| **TLS/HTTPS** | ❌ Not provided natively — SSH tunnel is encrypted, but the iPhone-to-VPS leg is unencrypted HTTP unless you add a proxy |

As with frp, the software is free but infrastructure costs apply. For someone who already operates a VPS, DIY SSH is effectively free.

### Linux Setup (Client Side)

```bash
# Generate a dedicated key for the tunnel:
ssh-keygen -t ed25519 -f ~/.ssh/idrsatunnel -N "" -C "reverse-tunnel"

# Copy the public key to your VPS:
scp ~/.ssh/idrsatunnel.pub user@your-vps-ip:/tmp/

# On the VPS, add the key to authorized_keys:
cat /tmp/idrsatunnel.pub >> ~/.ssh/authorized_keys

# Test the tunnel manually:
ssh -N -R 0.0.0.0:7000:localhost:8080 user@your-vps-ip \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3"

# On the VPS, ensure GatewayPorts is enabled in /etc/ssh/sshd_config:
# GatewayPorts clientspecified
```

**As a persistent systemd service (using autossh):**
```bash
sudo tee /etc/systemd/system/ssh-tunnel.service <<EOF
[Unit]
Description=Persistent SSH Reverse Tunnel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=youruser
Environment="AUTOSSHGATETIME=0"
ExecStart=/usr/bin/autossh -M 0 \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -o "ExitOnForwardFailure=yes" \
    -N -R 0.0.0.0:7000:localhost:8080 \
    -i ~/.ssh/idrsatunnel \
    user@your-vps-ip
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now ssh-tunnel.service
```

### iOS Access Method

- **Browser URL:** `http://<vps-ip>:7000` — open directly in Safari or Chrome. No app required.
- ⚠️ **No automatic HTTPS.** The traffic from iPhone to VPS is unencrypted HTTP (the SSH tunnel only encrypts VPS-to-Linux). For production use, configure nginx/Caddy on the VPS for TLS termination.

### Persistent vs. Ephemeral

**Excellent for persistent use**, assuming your VPS stays running. This approach has no artificial rate limits, bandwidth caps, or connection limits — it's limited only by your VPS resources and OpenSSH configuration. The main operational concern is tunnel resilience (addressed via autossh + systemd) and SSH key management. For a single-user always-on terminal access scenario with an existing VPS, this is the most transparent and cost-effective approach.

---

## Comparison Summary Table

| Method | Free? | Bandwidth | Custom Domain | WebSocket | iOS Access | Always-On Viability |
|---|---|---|---|---|---|---|
| **cloudflared** | ✅ Unlimited | ✅ None | ❌ Requires CF domain | ✅ Full | ✅ Browser URL | ⭐⭐⭐⭐⭐ Excellent |
| **ngrok** | ⚠️ 1 GB/mo | 5K TCP/mo | ❌ Auto-assigned only | ✅ (with header) | ✅ Browser URL | ⭐⭐ Fair (caps) |
| **Tailscale Funnel** | ✅ Free tier | ⚠️ Unspecified | `*.ts.net` format | ⚠️ Beta, relayed | ✅ Browser URL | ⭐⭐⭐ Good (beta risk) |
| **ZeroTier** | ✅ 10 devices | Not metered | ❌ Private IP only | ✅ On LAN | 📱 App required | ⭐⭐⭐ Good (private net) |
| **frp** | ✅ Free software | Unlimited* | Configurable | ✅ Full | ✅ Browser URL | ⭐⭐⭐⭐ Excellent* |
| **Serveo** | ✅ 3 tunnels | Not limited† | Custom subdomains | ⚠️ Via SSH | ✅ Browser URL | ⭐ Poor (reliability) |
| **LocalXpose** | ✅ Unlimited | 2 HTTP tunnels only | ❌ Auto-assigned only | ⚠️ Unverified on free | ✅ Browser URL | ⭐⭐⭐ Good (HTTP-only) |
| **DIY SSH tunnel** | ✅ Free software | Unlimited* | Configurable | ⚠️ Possible | ✅ Browser URL | ⭐⭐⭐⭐ Excellent* |

\* Requires a VPS (~$3–5/month infrastructure cost)
† Serveo's reliability is the concern, not bandwidth limits

---

## Final Recommendation

### Primary Recommendation: Cloudflare Tunnel (cloudflared)

**Cloudflared is the best choice for exposing an xterm.js web terminal from Linux to iPhone browsers.** The evidence supporting this recommendation:

1. **Unlimited bandwidth on the free tier** — Cloudflare's own documentation and community staff confirm there are no bandwidth limits for production tunnels. This is critical for a web terminal that may stream large amounts of output, handle file transfers, and maintain long-lived WebSocket connections.

2. **Full WebSocket support** — Unlike Quick Tunnels (which lack SSE support and have a 200-request cap), production tunnels fully support WebSockets without artificial limits. xterm.js relies on persistent WebSocket connections (`ws://` or `wss://`) for real-time terminal I/O, and cloudflared handles this transparently.

3. **No session expiry or connection limits** — Production tunnels are designed for always-on deployment. The daemon maintains four redundant connections to Cloudflare's edge and auto-reconnects on failures.

4. **Automatic HTTPS/TLS** — No manual certificate management required. The iPhone browser receives a valid TLS connection out of the box.

5. **Browser-accessible on iOS** — No native app needed. A public HTTPS URL opens directly in Safari or Chrome with full WebSocket support.

6. **No interstitial pages** — Unlike ngrok and Serveo, cloudflared does not inject warning pages between the user and the terminal.

7. **Outbound-only connectivity** — Only requires outbound HTTPS (port 443) from your Linux host. No inbound ports or firewall changes needed — ideal for environments where inbound traffic is blocked.

**Tradeoff:** You need a Cloudflare account and a domain hosted on Cloudflare DNS. However, you can use any existing domain (even one hosted elsewhere, by changing nameservers), and the Cloudflare free tier includes DNS hosting at no cost. For users without any domain, this is the only setup friction point — but it's still free.

### Alternative 1: Tailscale Funnel

**Best for:** Users who already have a Tailscale tailnet and want zero-configuration public exposure with strong privacy guarantees.

- ✅ Available on the free Personal plan (up to 100 devices)
- ✅ End-to-end encryption maintained through relay servers
- ✅ Browser-accessible via `*.ts.net` URLs — no iPhone app needed for Funnel access
- ⚠️ Still in beta — features and behavior may change
- ⚠️ Non-configurable bandwidth limits (no published cap, but not guaranteed unlimited)
- ⚠️ Port restrictions (only 443/8443/10000)

Tailscale Funnel is the strongest alternative when you already use Tailscale for device management. It requires less setup than cloudflared (no domain needed), but the beta status and unspecified bandwidth make it a secondary choice for production-critical always-on terminal access.

### Alternative 2: DIY SSH Reverse Tunneling (with existing VPS)

**Best for:** Users who already operate a VPS and want maximum control with zero recurring tunneling costs beyond the VPS itself.

- ✅ No third-party dependencies — just OpenSSH and autossh
- ✅ Unlimited bandwidth, no artificial limits
- ✅ Full transparency — you control every aspect of the tunnel
- ⚠️ Requires manual TLS setup on the VPS for HTTPS (nginx/Caddy)
- ⚠️ Operational overhead: VPS maintenance, key management, security hardening
- ⚠️ No automatic certificate provisioning

DIY SSH tunneling is the most "transparent" approach with the fewest hidden dependencies. It's ideal when you already have a VPS and want complete control over the tunnel lifecycle. However, it demands more operational knowledge than cloudflared's automated setup.

---

## Methods Not Recommended for This Use Case

- **ngrok:** The 1 GB/month transfer cap and 5,000 TCP connection limit are too restrictive for persistent terminal access. The interstitial page also adds friction for WebSocket-based apps.
- **Serveo:** Documented reliability issues since 2019 make it unsuitable for always-on deployment. The SSH-based architecture works in principle, but inconsistent uptime disqualifies it as a primary tool.
- **LocalXpose:** While the free tier offers unlimited bandwidth (more generous than ngrok), WebSocket support is unverified on the HTTP-only free plan. xterm.js requires persistent WebSocket connections that may not work reliably through an HTTP tunnel proxy. The 2-tunnel limit and lack of custom subdomains on the free tier also constrain always-on use.
- **ZeroTier:** While technically capable of exposing your terminal, ZeroTier requires both the Linux host and iPhone to join the same private network via a native app. This is fundamentally different from browser-accessible URL approaches and is better suited for trusted-device LAN scenarios rather than public internet exposure.
- **frp:** Functionally equivalent to DIY SSH tunneling in terms of capabilities and tradeoffs (requires a VPS, unlimited bandwidth). frp offers more features (dashboard, URL routing, authentication) but at the cost of additional software to manage. For a simple single-port terminal exposure, DIY SSH is simpler; for multi-service setups, frp may be preferable.

---

## Appendix: Quick-Start Commands Reference

### cloudflared (recommended — persistent systemd service)
```bash
# Install
sudo dpkg -i cloudflared-linux-amd64.deb

# Create tunnel and configure
cloudflared tunnel login
cloudflared tunnel create web-terminal

cat > /etc/cloudflared/config.yml <<EOF
tunnel: web-terminal
credentials-file: /root/.cloudflared/web-terminal.json

ingress:
  - hostname: terminal.example.com
    serviceName: http://localhost:8080
  - match: !<host> terminal.example.com
    service: http_error 404
EOF

sudo systemd-tmpfiles --create cloudflared.conf 2>/dev/null || true
sudo systemctl enable --now cloudflared-tunnel@web-terminal.service
```

### ngrok (persistent systemd service)
```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz && sudo cp ngrok /usr/local/bin/
ngrok config add-authtoken YOUR_TOKEN

cat > ~/.config/ngrok/ngrok.yml <<EOF
authtoken: YOUR_TOKEN
tunnels:
  web-terminal:
    addr: 8080
    proto: http
EOF

# Create systemd service referencing the config file
```

### Tailscale Funnel
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale funnel http://localhost:8080
# Access at: https://web-terminal.your-tailnet.ts.net
```

### frp (requires VPS)
```bash
# Client config (frpc.toml):
serverAddr = "your-vps-ip"
serverPort = 7000
auth.token = "secret"

[[proxies]]
name = "terminal"
type = "tcp"
localPort = 8080
remotePort = 8080
```

### Serveo (ephemeral)
```bash
ssh -R 80:localhost:8080 serveo.net
# For persistent: wrap with autossh
```

### DIY SSH Reverse Tunnel (requires VPS)
```bash
autossh -M 0 \
    -o "ServerAliveInterval=30" \
    -o "ServerAliveCountMax=3" \
    -N -R 0.0.0.0:7000:localhost:8080 \
    -i ~/.ssh/idrsatunnel \
    user@your-vps-ip
```

---

*Document prepared June 2026. All information based on publicly documented free-tier specifications and official documentation as of the publication date.*
