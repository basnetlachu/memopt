# MemOpt File Structure

## Production Files

```
memopt/
├── README.md                      # Project overview
├── setup.py                       # Python package setup
├── requirements.txt               # Core dependencies
│
├── memopt/                        # Core optimization library
│   ├── __init__.py
│   ├── model.py                   # OptimizedLLM class
│   ├── memory_manager.py
│   ├── attention.py
│   ├── kv_cache.py
│   └── ...                        # Other optimization modules
│
├── worker/                        # GPU Worker Service (Data Plane)
│   ├── main.py                    # Worker FastAPI service
│   ├── requirements.txt           # Worker dependencies
│   ├── Dockerfile                 # Worker container
│   ├── docker-compose.yml         # Worker deployment
│   └── .env.example               # Worker config template
│
├── saas/                          # Control Plane API
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Settings & validation
│   ├── database.py                # PostgreSQL models
│   ├── auth.py                    # API key authentication
│   ├── rate_limiter.py            # Rate limiting
│   └── requirements.txt           # API dependencies
│
├── tests/                         # Test suite
│   └── ...
│
├── Dockerfile                     # Control plane container (Coolify)
├── .env                           # Environment variables (gitignored)
│
└── Documentation
    ├── DEPLOYMENT.md              # Complete deployment guide
    ├── QUICKSTART.md              # Quick reference
    └── IMPLEMENTATION_SUMMARY.md  # Technical overview
```

## Key Files Explained

### Worker (GPU Server)

- **worker/main.py** - Secure inference service with token auth
- **worker/Dockerfile** - CUDA-enabled container (GPU or CPU)
- **worker/docker-compose.yml** - Deployment with GPU support
- **worker/.env** - Configuration (WORKER_TOKEN, MODEL_NAME)

### Control Plane (Coolify)

- **saas/main.py** - Public API (auth, billing, forwards to worker)
- **saas/config.py** - Environment validation
- **Dockerfile** - API container (no GPU needed)
- **.env** - Configuration (DATABASE_URL, WORKER_URL, WORKER_TOKEN)

### Documentation

- **DEPLOYMENT.md** - Step-by-step production deployment
- **QUICKSTART.md** - 10-minute quick start guide
- **IMPLEMENTATION_SUMMARY.md** - Technical architecture overview

## Deployment Locations

- **Control Plane**: Coolify VPS at memopt.sophisticatesai.com
- **Worker**: Your GPU server (internal network)
- **Database**: PostgreSQL on Coolify
