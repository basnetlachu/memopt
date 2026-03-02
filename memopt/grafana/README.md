# memopt Grafana Dashboard

9-panel dashboard for monitoring memopt GPU optimization in production.

## Panels

| # | Title | Source |
|---|-------|--------|
| 1 | GPU Utilisation % | `memopt_gpu_utilization_percent{gpu_id}` (api/server.py) |
| 2 | GPU Memory Used | `memopt_gpu_memory_used_bytes{gpu_id}` (api/server.py) |
| 3 | GPU Power Draw | `memopt_gpu_power_watts{gpu_id}` (api/server.py, pynvml 15s) |
| 4 | Optimisation Speedup | `memopt_speedup_ratio` histogram (api/server.py) |
| 5 | Optimisations Applied | `memopt_optimizations_applied_total` (api/server.py) |
| 6 | Dollar Savings / Hour | `memopt_dollar_saved_per_hour{node}` (zero_touch.py textfile) |
| 7 | Daemon Scan Rate | `memopt_total_scans{node}` (zero_touch.py textfile) |
| 8 | Active Drift Alerts | `memopt_drift_alert_active{node,model,severity}` (notifier.py textfile) |
| 9 | Drift Alert Rate | `memopt_drift_alerts_total{severity}` (notifier.py textfile) |

All metric names are sourced directly from the codebase — no synthetic names.

## Quick Start

### 1. node_exporter (textfile collector)

The daemon writes two `.prom` files to `~/.memopt/metrics/`:

```
daemon_metrics.prom   ← zero_touch.py
drift_alerts.prom     ← alerts/notifier.py
```

Start node_exporter pointing at that directory:

```bash
node_exporter \
  --collector.textfile.directory=$HOME/.memopt/metrics \
  --web.listen-address=:9100
```

### 2. memopt API metrics

The API server exports metrics at `:8000/metrics` via prometheus_client.
Add a Prometheus scrape job:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: memopt_api
    static_configs:
      - targets: ['localhost:8000']

  - job_name: memopt_node
    static_configs:
      - targets: ['localhost:9100']
```

### 3. Grafana provisioning (auto-load)

Copy provisioning files and dashboard JSON:

```bash
cp memopt/grafana/provisioning/datasources/prometheus.yml \
   /etc/grafana/provisioning/datasources/

cp memopt/grafana/provisioning/dashboards/memopt.yml \
   /etc/grafana/provisioning/dashboards/

cp memopt/grafana/memopt_dashboard.json \
   /etc/grafana/dashboards/

systemctl restart grafana-server
```

Dashboard loads automatically at **http://localhost:3000** under the title
**"memopt — GPU Optimization"**.

### 4. Manual import

In Grafana UI: **Dashboards → Import → Upload JSON file** →
select `memopt/grafana/memopt_dashboard.json`.
