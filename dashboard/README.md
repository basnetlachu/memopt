# Memopt Dashboard

Real-time GPU optimization monitoring across your entire fleet.

## Quick Start

### Docker Compose (Recommended)

```bash
cd dashboard
docker-compose up -d
```

Access the dashboard at http://localhost:3000

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GPU Server 1  │     │   GPU Server 2  │     │   GPU Server N  │
│  memopt daemon  │     │  memopt daemon  │     │  memopt daemon  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Dashboard Backend    │
                    │       (FastAPI)         │
                    │    - REST API           │
                    │    - WebSocket          │
                    │    - SQLite DB          │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Dashboard Frontend    │
                    │       (React)           │
                    │    - Fleet Overview     │
                    │    - Server Details     │
                    │    - Training Runs      │
                    │    - Session History    │
                    │    - Alerts             │
                    └─────────────────────────┘
```

## Connecting Daemons

Configure each GPU server's daemon to report to the dashboard:

**~/.memopt/daemon_config.yaml:**
```yaml
dashboard_url: http://dashboard-server:8000
dashboard_report_interval: 10.0
```

Or set the environment variable:
```bash
export MEMOPT_DASHBOARD_URL=http://dashboard-server:8000
memopt daemon start
```

## API Endpoints

### Fleet
- `GET /api/fleet/overview` - Aggregate fleet statistics

### Servers
- `GET /api/servers` - List all servers
- `POST /api/servers` - Register a server
- `GET /api/servers/{hostname}` - Get server details
- `POST /api/servers/{hostname}/heartbeat` - Update server metrics
- `GET /api/servers/{hostname}/metrics` - Get historical metrics

### Optimization Sessions
- `GET /api/sessions` - List sessions (with filtering)
- `POST /api/sessions` - Create session
- `GET /api/sessions/{id}` - Get session details
- `PATCH /api/sessions/{id}` - Update session

### Training Runs
- `GET /api/training` - List training runs
- `POST /api/training` - Create training run
- `PATCH /api/training/{id}` - Update training run

### Alerts
- `GET /api/alerts` - List alerts
- `POST /api/alerts` - Create alert
- `PATCH /api/alerts/{id}/acknowledge` - Acknowledge alert

### Export
- `GET /api/export/sessions?format=csv` - Export sessions

### WebSocket
- `WS /ws` - Real-time updates

## WebSocket Events

The WebSocket broadcasts these event types:

```javascript
// Server registered
{ "type": "server_registered", "server": { "id": 1, "hostname": "gpu-01" } }

// GPU metrics updated
{ "type": "gpu_update", "server": "gpu-01", "gpus": [...] }

// Optimization applied
{ "type": "optimization_applied", "session_id": "...", "model": "...", "speedup": 1.35 }

// Rollback triggered
{ "type": "rollback_triggered", "session_id": "...", "model": "..." }

// Training progress
{ "type": "training_progress", "run_id": "...", "epoch": 5, "total_epochs": 10, "loss": 0.123 }

// Alert
{ "type": "alert", "level": "warning", "title": "...", "message": "..." }
```

## Frontend Pages

1. **Fleet Overview** - Dashboard home with aggregate stats and server map
2. **Servers** - Detailed view of each GPU server with utilization charts
3. **Training Runs** - Active and recent training runs with progress
4. **Session History** - Searchable log of all optimization sessions
5. **Alerts** - System alerts and notifications

## Environment Variables

**Backend:**
- `DATABASE_URL` - SQLite connection string (default: `sqlite+aiosqlite:///./memopt_dashboard.db`)

**Daemon:**
- `MEMOPT_DASHBOARD_URL` - Dashboard API URL

## Production Deployment

For production, use the nginx profile:

```bash
docker-compose --profile production up -d
```

This adds nginx as a reverse proxy on port 80.

## Tech Stack

- **Backend:** FastAPI, SQLAlchemy, WebSockets
- **Frontend:** React, Recharts, TailwindCSS, Vite
- **Database:** SQLite (can be swapped for PostgreSQL)
- **Container:** Docker, Docker Compose
