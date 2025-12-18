# MemOpt Customer Onboarding Guide

**Goal: Get customer from "trial started" to "production deployment" in 14 days**

---

## Day 0: Contract Signed

### Immediate Actions
- [ ] Create customer Slack channel (#customer-[company])
- [ ] Add customer to GitHub repo (if Tier 2+)
- [ ] Send welcome email with credentials
- [ ] Schedule kickoff call (Day 1)
- [ ] Provision test environment if needed

### Welcome Email Template

```
Subject: Welcome to MemOpt - Let's Get You Saving Money

Hi [Name],

Welcome to MemOpt! We're excited to help you cut your inference costs in half.

Here's what happens next:

DAY 1 (Tomorrow): Kickoff call - 30 minutes
- Technical architecture review
- Installation walkthrough
- Set success metrics

DAY 3: First benchmark
- Run your models with MemOpt
- Measure baseline vs optimized
- Verify cost savings

DAY 7: Check-in call - 15 minutes
- Review results
- Troubleshoot any issues
- Plan production rollout

DAY 14: Production deployment
- Deploy to production
- Monitor metrics
- Celebrate savings!

You now have access to:
- Slack: [invite link]
- GitHub: [repo link]
- Documentation: docs.memopt.ai
- Support: support@memopt.ai (we respond in <4 hours)

Any questions before tomorrow? Just reply to this email.

Looking forward to working with you!

[Your Name]
MemOpt Customer Success
```

---

## Day 1: Kickoff Call (30 minutes)

### Agenda

1. **Introductions (5 min)**
   - Customer team: eng lead, infra lead, product lead
   - MemOpt team: customer success, solutions engineer

2. **Technical Discovery (10 min)**
   - Which models? (LLaMA, Mistral, custom?)
   - What scale? (tokens/day, requests/sec)
   - What hardware? (A100, H100, count)
   - What deployment? (AWS, GCP, Azure, on-prem)
   - What frameworks? (HuggingFace, vLLM, custom)
   - What metrics matter? (latency, throughput, cost)

3. **Installation Plan (10 min)**
   - Review hardware requirements
   - Walk through installation steps
   - Test on one GPU first
   - Set up monitoring

4. **Success Metrics (5 min)**
   - Define "success" for trial
   - Typically: 40%+ cost reduction, <1% accuracy loss
   - Set measurement methodology
   - Agree on production rollout criteria

### Action Items from Call
- [ ] Customer: Install MemOpt on test GPU
- [ ] Customer: Share sample prompts for testing
- [ ] MemOpt: Provide model-specific optimization guide
- [ ] Both: Schedule Day 3 check-in

---

## Day 1-2: Installation

### Customer Self-Service Steps

```bash
# 1. Set up environment
conda create -n memopt python=3.10
conda activate memopt

# 2. Install dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate

# 3. Install MemOpt
pip install memopt

# 4. Verify installation
python -c "from memopt import OptimizedLLM; print('Success!')"

# 5. Run quick test
python example.py basic
```

### Common Installation Issues

**Issue: CUDA version mismatch**
```bash
# Check CUDA version
nvidia-smi

# Install correct PyTorch
# For CUDA 11.8:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**Issue: Out of memory**
```python
# Use smaller model for testing first
model = OptimizedLLM("meta-llama/Llama-2-7b-hf")  # Start with 7B

# Or use CPU offloading
model = OptimizedLLM(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto"  # Automatic CPU offloading
)
```

**Issue: Model not found**
```bash
# Make sure you have access to the model
huggingface-cli login

# Or use local model path
model = OptimizedLLM("/path/to/local/model")
```

---

## Day 3: First Benchmark

### Run Customer's Workload

```bash
# Baseline benchmark
python benchmark.py \
  --model [customer-model] \
  --mode baseline \
  --num-prompts 20 \
  --max-tokens 512

# Optimized benchmark
python benchmark.py \
  --model [customer-model] \
  --mode optimized \
  --optimization-level high \
  --num-prompts 20 \
  --max-tokens 512

# Compare both
python benchmark.py --model [customer-model] --mode both
```

### Analyze Results

**What to look for:**
- Throughput improvement: Should be 2-3x
- Memory reduction: Should be 40-60%
- Cost reduction: Should be 50%+
- GPU stall: Should drop from ~75% to ~35%

**If results are lower than expected:**
1. Check GPU utilization: `nvidia-smi dmon`
2. Verify model is on GPU: Check memory usage
3. Try different optimization level: `aggressive`
4. Check for batching opportunities

### Report to Customer

```
Subject: Your MemOpt Benchmark Results

Hi [Name],

Great news - we just ran your workload through MemOpt. Here are your results:

BASELINE (Current State):
- Throughput: [X] tok/s
- Memory: [Y] GB
- Cost per 1M tokens: $[Z]

WITH MEMOPT:
- Throughput: [X2] tok/s (+[%]%)
- Memory: [Y2] GB (-[%]%)
- Cost per 1M tokens: $[Z2] (-[%]%)

At your scale ([tokens] tokens/day), this translates to:
- Daily savings: $[daily]
- Annual savings: $[annual]
- Payback: [days] days

Next steps:
1. Review these numbers
2. Test on more of your workloads
3. If satisfied, plan production rollout

Questions? Let's jump on a quick call.

[Your Name]
```

---

## Day 3-7: Extended Testing

### Customer Testing Checklist

- [ ] Test on all production models
- [ ] Run accuracy benchmarks
- [ ] Test with real user prompts
- [ ] Measure latency at different loads
- [ ] Test error handling
- [ ] Verify monitoring integration

### Accuracy Verification

```python
from memopt import OptimizedLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load baseline model
baseline = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Load optimized model
optimized = OptimizedLLM("your-model", optimization_level="high")

# Test prompts
test_prompts = [
    # Your test set here
]

# Compare outputs
for prompt in test_prompts:
    baseline_output = baseline.generate(...)
    optimized_output = optimized.generate(...)
    
    # Calculate similarity (BLEU, ROUGE, etc.)
    # Should be >99% similar
```

### Load Testing

```python
import time
import concurrent.futures

def inference_request(prompt):
    model = OptimizedLLM("your-model")
    return model.generate(prompt, max_tokens=256)

# Simulate concurrent requests
prompts = ["test prompt"] * 100

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(inference_request, prompts))
end = time.time()

print(f"Total time: {end - start:.2f}s")
print(f"Throughput: {len(prompts) / (end - start):.1f} req/s")
```

---

## Day 7: Check-in Call (15 minutes)

### Agenda

1. **Results Review (5 min)**
   - What metrics did you see?
   - Any surprises?
   - Accuracy acceptable?

2. **Issues & Questions (5 min)**
   - Any problems?
   - Performance concerns?
   - Integration questions?

3. **Production Plan (5 min)**
   - Ready to deploy?
   - Rollout strategy (canary vs full)
   - Monitoring setup
   - Rollback plan

### Decision Point

**If satisfied:** Plan production deployment for Day 14
**If not satisfied:** Identify blockers, extend trial, provide more support
**If terrible results:** Diagnose issues, possibly not a good fit

---

## Day 7-14: Production Preparation

### Production Deployment Checklist

- [ ] Set up production environment
- [ ] Install MemOpt on all inference GPUs
- [ ] Configure monitoring (Prometheus, Datadog, etc.)
- [ ] Set up alerting
- [ ] Create rollback plan
- [ ] Test failover scenarios
- [ ] Document runbooks
- [ ] Train ops team

### Monitoring Setup

```python
from memopt import OptimizedLLM
from prometheus_client import Counter, Histogram, Gauge

# Set up metrics
inference_counter = Counter('memopt_inferences_total', 'Total inferences')
latency_histogram = Histogram('memopt_latency_seconds', 'Inference latency')
memory_gauge = Gauge('memopt_memory_gb', 'GPU memory usage')

# Use in production
model = OptimizedLLM("your-model", enable_profiling=True)

def handle_inference(prompt):
    with latency_histogram.time():
        response = model.generate(prompt, max_tokens=256)
        inference_counter.inc()
        
        stats = model.get_profiling_stats()
        memory_gauge.set(stats.peak_memory_allocated_gb)
    
    return response
```

### Rollout Strategy

**Option 1: Canary Deployment (Recommended)**
- Deploy to 10% of traffic
- Monitor for 24 hours
- If good, roll out to 50%
- Monitor for 24 hours
- If good, roll out to 100%

**Option 2: Blue-Green Deployment**
- Deploy to separate infrastructure
- Switch traffic all at once
- Keep old infrastructure as backup
- Decommission after 1 week

---

## Day 14: Production Deployment

### Deployment Day Checklist

**Pre-deployment (Morning)**
- [ ] Verify all tests passing
- [ ] Confirm monitoring working
- [ ] Have rollback plan ready
- [ ] Customer engineering team on standby
- [ ] MemOpt team on standby

**Deployment (Afternoon)**
- [ ] Deploy to canary (10%)
- [ ] Monitor for 1 hour
- [ ] Check key metrics
- [ ] Verify no errors
- [ ] Deploy to 50%
- [ ] Monitor for 2 hours
- [ ] Deploy to 100%

**Post-deployment (Evening)**
- [ ] Monitor overnight
- [ ] Check morning metrics
- [ ] Verify cost savings
- [ ] Customer happiness check

### Launch Email Template

```
Subject: 🚀 MemOpt is Live - You're Saving Money!

Hi [Name],

Congratulations! MemOpt is now live in your production environment.

PRODUCTION METRICS (First 24 hours):
- Throughput: [X] tok/s
- Cost per 1M tokens: $[Y]
- GPU utilization: [Z]%

ESTIMATED SAVINGS:
- Daily: $[amount]
- Monthly: $[amount]
- Annual: $[amount]

Your monitoring dashboard: [link]

What's next:
- We'll monitor closely for the first week
- Weekly check-ins for the first month
- Then monthly check-ins

Questions or issues? We're here 24/7.

Congrats on the launch!

[Your Name]
```

---

## Week 2-4: Stabilization

### Weekly Check-ins

**Week 2:**
- Review first week metrics
- Identify any issues
- Optimize further if needed

**Week 3:**
- Metrics stable?
- Customer satisfied?
- Any feature requests?

**Week 4:**
- Move to monthly check-ins
- Request testimonial
- Ask for referrals

---

## Success Criteria

### Technical Success
- ✅ 40%+ cost reduction
- ✅ <1% accuracy degradation
- ✅ 95%+ uptime
- ✅ Latency within SLA

### Business Success
- ✅ Customer renews
- ✅ Customer provides testimonial
- ✅ Customer refers other companies
- ✅ Customer upgrades tier

### Red Flags
- ❌ Cost reduction <20%
- ❌ Accuracy loss >2%
- ❌ Customer not engaged
- ❌ Multiple production issues

---

## Common Issues & Solutions

### "We're not seeing the expected speedup"

**Diagnose:**
```bash
# Check GPU utilization
nvidia-smi dmon -s u -c 10

# Check memory bandwidth
nvidia-smi dmon -s m -c 10

# Profile code
python -m torch.profiler [your script]
```

**Common causes:**
- CPU bottleneck (check with `htop`)
- Disk I/O bottleneck (model loading)
- Network bottleneck (distributed inference)
- Small batch size (increase batching)

### "Accuracy is lower than expected"

**Solutions:**
1. Reduce quantization: `optimization_level="balanced"` → `"conservative"`
2. Run accuracy benchmark on specific tasks
3. Fine-tune quantization thresholds
4. Consider hybrid approach (some layers FP16)

### "Getting CUDA out of memory errors"

**Solutions:**
```python
# Option 1: Smaller blocks
model = OptimizedLLM(
    "your-model",
    kv_block_size=8  # Default is 16
)

# Option 2: CPU offloading
model = OptimizedLLM(
    "your-model",
    device_map="auto"
)

# Option 3: Gradient checkpointing
model = OptimizedLLM(
    "your-model",
    use_gradient_checkpointing=True
)
```

---

## Escalation Path

**Tier 1 Support (Customer Success):**
- Usage questions
- Best practices
- Integration help
- Monitoring setup

**Tier 2 Support (Solutions Engineer):**
- Performance optimization
- Custom configurations
- Multi-GPU setup
- Advanced features

**Tier 3 Support (Core Engineering):**
- Bugs
- Feature requests
- Custom model support
- Architecture changes

**Escalate to Tier 2 if:**
- Customer blocked for >4 hours
- Performance <50% of expected
- Custom requirements
- Multi-GPU needed

**Escalate to Tier 3 if:**
- Potential bug in MemOpt
- Model incompatibility
- Need core changes
- Research required

---

## Customer Health Score

Track monthly:

| Metric | Points | Your Score |
|--------|---------|------------|
| Cost reduction >40% | 25 | |
| Production deployment | 25 | |
| No critical issues | 20 | |
| Response to check-ins | 15 | |
| Using advanced features | 10 | |
| Provided testimonial | 5 | |

**Score interpretation:**
- 90-100: Healthy, likely renewal
- 70-89: At risk, needs attention
- <70: Critical, escalate to sales

---

## Post-Trial Actions

### If Converted to Paid
- [ ] Send thank you + invoice
- [ ] Request LinkedIn recommendation
- [ ] Ask for case study
- [ ] Add to reference customer list
- [ ] Quarterly business review

### If Not Converted
- [ ] Exit interview (understand why)
- [ ] Offer extended trial if close
- [ ] Keep in touch for future
- [ ] Learn from feedback

---

## Remember

- **Speed matters:** 14 days trial → production
- **Be proactive:** Weekly check-ins, don't wait for problems
- **Show value:** Constant reminders of savings
- **Build relationship:** They should text you with questions
- **Document everything:** Every issue makes your support better

**Goal: 80% trial-to-customer conversion rate**
