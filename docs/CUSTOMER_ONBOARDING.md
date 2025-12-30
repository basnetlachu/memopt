# MemOpt Customer Onboarding Guide

**For MemOpt Sales/Support Team**

This document describes the complete customer onboarding process from contract signing to production deployment.

## Pre-Requisites

Before onboarding customers, ensure you have:

- ✅ Generated Ed25519 keypair (one time only)
- ✅ Updated `PUBLIC_KEY_HEX` in `memopt/license.py`
- ✅ Updated `PRIVATE_KEY_HEX` in `tools/sign_license.py`
- ✅ Built MemOpt package or published to PyPI

**Validation:**
```bash
python tools/preflight_check.py
```

---

## Step 1: Contract Signed

Customer has signed either:

### Enterprise Flat ($800k/year)
- Fixed annual fee
- Up to 16 GPUs
- All features included
- No telemetry required
- Enterprise support (24/7, Slack, email)

### Revenue Share (35% of savings)
- Pay based on GPU cost savings
- Unlimited GPUs
- All features included
- Monthly telemetry required for billing
- Standard support (email, GitHub issues)

**Action:** Collect customer details:
- Company name
- Technical contact (name, email)
- Number of GPUs
- License tier
- License duration (typically 1 year)

---

## Step 2: Generate Customer License

### Using the License Signing Tool

```bash
# Enterprise customer example
python tools/sign_license.py \
    --customer-id acme-corp \
    --tier enterprise \
    --expires 2026-12-30 \
    --max-gpus 8 \
    --features vllm,quantization,profiling

# Output: license_acme-corp.json
```

### License Parameters

**Customer ID:**
- Format: lowercase, hyphens (e.g., `acme-corp`, `startup-inc`)
- Must be unique
- Used in logs and support tickets

**Tier:**
- `enterprise` - Flat fee, no telemetry
- `revenue_share` - Usage-based, requires telemetry

**Expires:**
- Format: `YYYY-MM-DD`
- Typically 1 year from issue date
- Customer gets warning 30 days before expiration

**Max GPUs:**
- Enterprise: Actual count (e.g., 8, 16)
- Revenue Share: 999 (unlimited)

**Features:**
- Always include: `vllm,quantization,profiling`
- Future features: `multi_gpu`, `telemetry_dashboard`

### Example Licenses

**Enterprise Customer:**
```json
{
  "customer_id": "acme-corp",
  "tier": "enterprise",
  "issued_at": "2025-12-30T00:00:00Z",
  "expires_at": "2026-12-30T23:59:59Z",
  "max_gpus": 8,
  "features": ["vllm", "quantization", "profiling"],
  "telemetry_required": false,
  "signature": "base64-encoded-signature"
}
```

**Revenue Share Customer:**
```json
{
  "customer_id": "startup-inc",
  "tier": "revenue_share",
  "issued_at": "2025-12-30T00:00:00Z",
  "expires_at": "2026-12-30T23:59:59Z",
  "max_gpus": 999,
  "features": ["vllm", "quantization", "profiling"],
  "telemetry_required": true,
  "signature": "base64-encoded-signature"
}
```

---

## Step 3: Send Welcome Package to Customer

**Email Template:**

```
Subject: Welcome to MemOpt - Your License & Installation Guide

Hi [Customer Name],

Welcome to MemOpt! We're excited to help you optimize your vLLM deployment.

Attached:
1. license.json - Your MemOpt license file
2. MemOpt_Installation_Guide.pdf - Step-by-step installation

Quick Start:
```bash
# Install MemOpt
pip install memopt

# Place license
sudo mkdir -p /etc/memopt
sudo cp license.json /etc/memopt/
sudo chmod 644 /etc/memopt/license.json

# Enable MemOpt
export MEMOPT_ENABLED=1
export MEMOPT_LICENSE_PATH=/etc/memopt/license.json

# Restart vLLM (MemOpt auto-attaches)
python -m vllm.entrypoints.openai.api_server --model /models/llama
```

Validation:
```bash
memopt doctor
```

Support:
- Documentation: https://docs.memopt.ai
- Email: support@memopt.ai
[Enterprise only: - Slack: #memopt-acme-corp]

Best regards,
MemOpt Team
```

**Attachments:**
1. `license_[customer-id].json` - Generated license file
2. Installation PDF (create from docs/ONPREM_VLLM_PLUGIN.md)

---

## Step 4: Customer Installation

### Customer's Installation Steps

**1. Install MemOpt Package**

```bash
# From PyPI (recommended)
pip install memopt

# Or from wheel file (if sent directly)
pip install memopt-0.1.0-py3-none-any.whl
```

**2. Place License File**

```bash
sudo mkdir -p /etc/memopt
sudo cp license.json /etc/memopt/
sudo chmod 644 /etc/memopt/license.json
```

**3. Configure Environment**

Add to `~/.bashrc` or deployment scripts:

```bash
export MEMOPT_ENABLED=1
export MEMOPT_LICENSE_PATH=/etc/memopt/license.json
```

**Optional:**
```bash
export MEMOPT_SAFE_MODE=1    # First deployment (static opts only)
export MEMOPT_STRICT=1       # Fail if patches don't apply
```

**4. Restart vLLM**

```bash
# MemOpt will auto-initialize when vLLM starts
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --port 8000
```

**Expected Output:**
```
[MemOpt] License validated for customer: acme-corp
[MemOpt] Detected vLLM version: 0.3.2
[MemOpt] Applying vLLM patches...
[MemOpt] Successfully patched: scheduler, kv_cache, memory_planner
[MemOpt] vLLM plugin initialized successfully
INFO:     Started server process [12345]
```

**5. Validate Installation**

```bash
memopt doctor
```

**Expected:**
```
============================================================
License Status
============================================================
Customer ID: acme-corp
Tier: enterprise
Status: VALID

============================================================
Summary
============================================================
Status: ALL SYSTEMS OPERATIONAL
```

---

## Step 5: First Week Support

### Day 1: Installation Verification

**Action Items:**
1. Confirm customer received email
2. Check for installation questions
3. Ask for `memopt doctor` output

**Common Issues:**

**License not found:**
```bash
# Solution
export MEMOPT_LICENSE_PATH=/etc/memopt/license.json
```

**Invalid signature:**
- Verify license file integrity
- Regenerate license if corrupted
- Check PUBLIC_KEY_HEX matches in deployed code

**vLLM not detected:**
```bash
pip install vllm
```

### Day 3: Performance Validation

**Ask customer to report:**
1. Memory usage (before/after MemOpt)
2. Throughput (requests/sec)
3. Any errors in logs

**Expected Results:**
- Memory: -40% to -60%
- Throughput: Neutral to +30%
- P99 latency: Neutral to -10%

### Day 7: Check-in Call

**Agenda:**
1. Review performance improvements
2. Address any issues
3. Discuss advanced features
4. Collect feedback

---

## Step 6: Ongoing Support

### Enterprise Tier

**Channels:**
- Dedicated Slack channel
- Email: support@memopt.ai
- 24/7 on-call for critical issues

**SLA:**
- Critical issues: 1 hour response
- High priority: 4 hour response
- Normal: 24 hour response

### Revenue Share Tier

**Channels:**
- Email: support@memopt.ai
- GitHub Issues

**SLA:**
- Critical issues: 4 hour response
- Normal: 48 hour response

---

## Troubleshooting Playbook

### License Issues

**Error: License expired**
```bash
# Generate renewal
python tools/sign_license.py \
    --customer-id [same-id] \
    --tier [same-tier] \
    --expires [new-date] \
    --max-gpus [same-count] \
    --features vllm,quantization,profiling

# Send to customer
```

**Error: GPU limit exceeded**
```bash
# Customer needs upgrade
# Generate new license with higher max-gpus
```

### Performance Issues

**Throughput decreased**
```bash
# Try safe mode
export MEMOPT_SAFE_MODE=1

# Check vLLM version compatibility
memopt doctor
```

**High latency**
```bash
# Disable MemOpt temporarily to isolate
export MEMOPT_ENABLED=0

# Compare metrics
```

### Integration Issues

**vLLM version not supported**
```bash
# Check compatibility
memopt doctor

# Use safe mode (static config only)
export MEMOPT_SAFE_MODE=1
```

---

## License Renewal Process

**30 Days Before Expiration:**

1. Email customer with renewal notice
2. Confirm GPU count, tier unchanged
3. Generate new license with extended date
4. Send updated license file

**Customer Action:**
```bash
# Replace license file
sudo cp new_license.json /etc/memopt/license.json

# Restart vLLM
```

**No downtime required** - hot reload on restart

---

## Offboarding (Contract Ends)

**Action:**
1. Do NOT generate renewal license
2. Customer's MemOpt stops working after expiration
3. vLLM continues running normally (MemOpt fails to initialize)
4. No data loss, no downtime

**Customer Experience:**
```
[MemOpt] License expired on 2026-12-30T23:59:59Z
[MemOpt] vLLM will run normally without optimizations
INFO:     Started server process [12345]
```

---

## Internal Checklist

**For Each New Customer:**

- [ ] Contract signed and payment confirmed
- [ ] Customer details collected
- [ ] License generated: `python tools/sign_license.py`
- [ ] License validated: Check signature
- [ ] Welcome email sent with license + docs
- [ ] Slack channel created (Enterprise only)
- [ ] Added to customer tracking spreadsheet
- [ ] Day 1 follow-up scheduled
- [ ] Day 3 check-in scheduled
- [ ] Day 7 call scheduled
- [ ] Renewal reminder set (11 months)

**Tools:**
- Customer tracker: Google Sheets
- License storage: Secure vault (1Password, etc.)
- Support tickets: Zendesk / GitHub Issues

---

## Metrics to Track

**Per Customer:**
- Installation date
- vLLM version
- GPU count
- Memory savings (%)
- Throughput improvement (%)
- Support tickets opened
- License renewal date

**Aggregate:**
- Total active licenses
- Average memory savings
- Average throughput improvement
- Support ticket volume
- Renewal rate

---

## Quick Reference

**Generate License:**
```bash
python tools/sign_license.py --customer-id [id] --tier [tier] --expires [date] --max-gpus [count] --features vllm,quantization,profiling
```

**Validate Setup:**
```bash
python tools/preflight_check.py
```

**Test License:**
```bash
python tools/sign_license.py --test
```

**Customer Installation:**
```bash
pip install memopt
sudo cp license.json /etc/memopt/
export MEMOPT_ENABLED=1
memopt doctor
```

---

**Questions?** Contact internal team lead or engineering@memopt.ai
