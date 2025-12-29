# MemOpt SaaS API Usage Examples

Complete examples for using the MemOpt SaaS API.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Admin Operations](#admin-operations)
3. [Inference API](#inference-api)
4. [Usage & Billing](#usage--billing)
5. [Python Client Examples](#python-client-examples)
6. [JavaScript Client Examples](#javascript-client-examples)
7. [curl Examples](#curl-examples)

---

## Quick Start

### 1. Get Your API Key

Your API key is provided when your tenant is created. It looks like:
```
sk_memopt_abc123xyz789...
```

**Store it securely - it's only shown once!**

### 2. Test Connection

```bash
curl https://api.memopt.io/health
```

---

## Admin Operations

### Create Enterprise Tenant ($800k/year flat fee)

```bash
curl -X POST https://api.memopt.io/admin/tenants \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Acme Corporation",
    "tier": "enterprise_flat",
    "allowed_models": null,
    "max_context_length": 8192,
    "priority_level": 10
  }'
```

Response:
```json
{
  "tenant_id": 1,
  "name": "Acme Corporation",
  "tier": "enterprise_flat",
  "api_key": "sk_memopt_XYZ123..."
}
```

### Create Revenue Share Tenant (35% of savings)

```bash
curl -X POST https://api.memopt.io/admin/tenants \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Startup XYZ",
    "tier": "revenue_share",
    "allowed_models": "gpt2,gpt2-medium,gpt2-large",
    "max_context_length": 2048,
    "priority_level": 5
  }'
```

### Suspend Tenant

```bash
curl -X POST https://api.memopt.io/admin/tenants/1/suspend \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

### Generate New API Key for Tenant

```bash
curl -X POST https://api.memopt.io/admin/tenants/1/api-keys \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "Production Key 2024-01"}'
```

### Revoke API Key

```bash
curl -X DELETE https://api.memopt.io/admin/api-keys/5 \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

### Generate Monthly Invoices (All Tenants)

```bash
curl -X POST "https://api.memopt.io/admin/invoices/generate?year=2024&month=1" \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

---

## Inference API

### Basic Inference Request

```bash
curl -X POST https://api.memopt.io/v1/infer \
  -H "X-API-Key: sk_memopt_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 256,
    "model": "gpt2",
    "optimization_level": "high"
  }'
```

Response:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "generated_text": "Machine learning is a subset of artificial intelligence...",
  "prompt_tokens": 5,
  "completion_tokens": 150,
  "total_tokens": 155,
  "latency_ms": 1250.5,
  "model": "gpt2"
}
```

### With Custom Parameters

```bash
curl -X POST https://api.memopt.io/v1/infer \
  -H "X-API-Key: sk_memopt_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about programming:",
    "max_tokens": 50,
    "temperature": 0.9,
    "top_p": 0.95,
    "model": "gpt2-medium",
    "optimization_level": "maximum"
  }'
```

### Different Optimization Levels

**Conservative** (safer, less aggressive optimizations):
```bash
curl -X POST https://api.memopt.io/v1/infer \
  -H "X-API-Key: sk_memopt_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain neural networks",
    "optimization_level": "conservative"
  }'
```

**High** (recommended, best balance):
```bash
{
  "optimization_level": "high"
}
```

**Maximum** (most aggressive, highest speedup):
```bash
{
  "optimization_level": "maximum"
}
```

---

## Usage & Billing

### Get Current Month Usage

```bash
curl https://api.memopt.io/v1/usage \
  -H "X-API-Key: sk_memopt_YOUR_KEY"
```

Response:
```json
{
  "total_requests": 15000,
  "total_tokens": 24000000,
  "prompt_tokens": 8000000,
  "completion_tokens": 16000000,
  "avg_latency_ms": 1180.5,
  "failed_requests": 12,
  "success_rate": 99.92
}
```

### Get Usage for Specific Date Range

```bash
curl "https://api.memopt.io/v1/usage?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z" \
  -H "X-API-Key: sk_memopt_YOUR_KEY"
```

### List All Invoices

```bash
curl https://api.memopt.io/v1/invoices \
  -H "X-API-Key: sk_memopt_YOUR_KEY"
```

Response:
```json
[
  {
    "id": 1,
    "tenant_id": 1,
    "period_start": "2024-01-01T00:00:00Z",
    "period_end": "2024-02-01T00:00:00Z",
    "tier": "revenue_share",
    "total_tokens": 50000000,
    "total_requests": 100000,
    "amount_usd": 100.00,
    "status": "issued",
    "created_at": "2024-02-01T00:00:00Z",
    "issued_at": "2024-02-01T00:00:00Z",
    "paid_at": null,
    "notes": "Usage: 100,000 requests, 50,000,000 tokens"
  }
]
```

### Get Specific Invoice

```bash
curl https://api.memopt.io/v1/invoices/1 \
  -H "X-API-Key: sk_memopt_YOUR_KEY"
```

---

## Python Client Examples

### Simple Client

```python
import requests

class MemOptClient:
    def __init__(self, api_key: str, base_url: str = "https://api.memopt.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def infer(self, prompt: str, **kwargs):
        """Run inference"""
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "model": kwargs.get("model", "gpt2"),
            "optimization_level": kwargs.get("optimization_level", "high")
        }

        response = requests.post(
            f"{self.base_url}/v1/infer",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_usage(self, start=None, end=None):
        """Get usage statistics"""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = requests.get(
            f"{self.base_url}/v1/usage",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def list_invoices(self):
        """List all invoices"""
        response = requests.get(
            f"{self.base_url}/v1/invoices",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = MemOptClient(api_key="sk_memopt_YOUR_KEY")

# Run inference
result = client.infer(
    prompt="What is quantum computing?",
    max_tokens=500,
    optimization_level="high"
)

print(f"Generated: {result['generated_text']}")
print(f"Tokens used: {result['total_tokens']}")
print(f"Latency: {result['latency_ms']}ms")

# Get usage
usage = client.get_usage()
print(f"Total tokens this month: {usage['total_tokens']:,}")
print(f"Success rate: {usage['success_rate']}%")

# List invoices
invoices = client.list_invoices()
for invoice in invoices:
    print(f"Invoice #{invoice['id']}: ${invoice['amount_usd']} ({invoice['status']})")
```

### Async Client (aiohttp)

```python
import aiohttp
import asyncio

class AsyncMemOptClient:
    def __init__(self, api_key: str, base_url: str = "https://api.memopt.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    async def infer(self, prompt: str, **kwargs):
        """Run inference asynchronously"""
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
            "model": kwargs.get("model", "gpt2"),
            "optimization_level": kwargs.get("optimization_level", "high")
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/infer",
                json=payload,
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()

# Usage
async def main():
    client = AsyncMemOptClient(api_key="sk_memopt_YOUR_KEY")

    # Run multiple inferences in parallel
    tasks = [
        client.infer("What is AI?"),
        client.infer("Explain machine learning"),
        client.infer("What is deep learning?")
    ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"Request {i+1}: {result['total_tokens']} tokens, {result['latency_ms']}ms")

asyncio.run(main())
```

---

## JavaScript Client Examples

### Node.js Client

```javascript
const axios = require('axios');

class MemOptClient {
    constructor(apiKey, baseURL = 'https://api.memopt.io') {
        this.apiKey = apiKey;
        this.baseURL = baseURL;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }

    async infer(prompt, options = {}) {
        const payload = {
            prompt,
            max_tokens: options.maxTokens || 256,
            temperature: options.temperature || 1.0,
            top_p: options.topP || 1.0,
            model: options.model || 'gpt2',
            optimization_level: options.optimizationLevel || 'high'
        };

        const response = await axios.post(
            `${this.baseURL}/v1/infer`,
            payload,
            { headers: this.headers }
        );

        return response.data;
    }

    async getUsage(start = null, end = null) {
        const params = {};
        if (start) params.start = start;
        if (end) params.end = end;

        const response = await axios.get(
            `${this.baseURL}/v1/usage`,
            { headers: this.headers, params }
        );

        return response.data;
    }

    async listInvoices() {
        const response = await axios.get(
            `${this.baseURL}/v1/invoices`,
            { headers: this.headers }
        );

        return response.data;
    }
}

// Usage
(async () => {
    const client = new MemOptClient('sk_memopt_YOUR_KEY');

    try {
        // Run inference
        const result = await client.infer(
            'What is quantum computing?',
            { maxTokens: 500, optimizationLevel: 'high' }
        );

        console.log(`Generated: ${result.generated_text}`);
        console.log(`Tokens: ${result.total_tokens}`);
        console.log(`Latency: ${result.latency_ms}ms`);

        // Get usage
        const usage = await client.getUsage();
        console.log(`Total tokens: ${usage.total_tokens.toLocaleString()}`);

        // List invoices
        const invoices = await client.listInvoices();
        invoices.forEach(invoice => {
            console.log(`Invoice #${invoice.id}: $${invoice.amount_usd} (${invoice.status})`);
        });
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
})();
```

### Browser (Fetch API)

```javascript
class MemOptClient {
    constructor(apiKey, baseURL = 'https://api.memopt.io') {
        this.apiKey = apiKey;
        this.baseURL = baseURL;
    }

    async infer(prompt, options = {}) {
        const response = await fetch(`${this.baseURL}/v1/infer`, {
            method: 'POST',
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt,
                max_tokens: options.maxTokens || 256,
                model: options.model || 'gpt2',
                optimization_level: options.optimizationLevel || 'high'
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
    }
}

// Usage in browser
const client = new MemOptClient('sk_memopt_YOUR_KEY');

document.getElementById('submit').addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value;

    try {
        const result = await client.infer(prompt);
        document.getElementById('output').textContent = result.generated_text;
        document.getElementById('stats').textContent =
            `Tokens: ${result.total_tokens}, Latency: ${result.latency_ms}ms`;
    } catch (error) {
        console.error('Error:', error);
    }
});
```

---

## curl Examples

### Error Handling

```bash
# Invalid API key
curl -X POST https://api.memopt.io/v1/infer \
  -H "X-API-Key: sk_memopt_INVALID" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'

# Response: 401 Unauthorized
{
  "detail": "Invalid API key"
}
```

```bash
# Rate limit exceeded
# Response: 429 Too Many Requests
{
  "detail": "Rate limit exceeded: 60 requests per minute"
}
```

```bash
# Model not allowed
curl -X POST https://api.memopt.io/v1/infer \
  -H "X-API-Key: sk_memopt_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "test",
    "model": "gpt2-xl"
  }'

# Response: 403 Forbidden (if model not in allowed_models)
{
  "detail": "Model 'gpt2-xl' not allowed for your tier"
}
```

### Using Authorization Header (Alternative)

```bash
# Instead of X-API-Key, use Authorization: Bearer
curl -X POST https://api.memopt.io/v1/infer \
  -H "Authorization: Bearer sk_memopt_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?"
  }'
```

---

## Rate Limits

### Revenue Share Tier
- **Requests**: 60 per minute
- **Tokens**: 100,000 per minute
- **Concurrent**: 5 requests
- **Monthly quota**: 100,000,000 tokens

### Enterprise Tier
- **Requests**: 10,000 per minute
- **Tokens**: 10,000,000 per minute
- **Concurrent**: 100 requests
- **Monthly quota**: Unlimited

When rate limited, wait for the period specified in `Retry-After` header.

---

## Best Practices

1. **Store API keys securely** - use environment variables
2. **Handle rate limits** - implement exponential backoff
3. **Log request IDs** - for debugging and support
4. **Monitor usage** - check `/v1/usage` regularly
5. **Use appropriate optimization levels** - `high` for most cases
6. **Cache results** - avoid redundant requests
7. **Handle errors gracefully** - retry transient failures

---

## Support

For API issues or questions:
- Check the API docs: `https://api.memopt.io/docs`
- Contact support with your `request_id` for faster resolution
