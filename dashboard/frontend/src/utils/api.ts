const API_BASE = '/api'

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }

  return response.json()
}

export interface FleetOverview {
  total_servers: number
  online_servers: number
  total_gpus: number
  total_memory_gb: number
  total_speedup: number
  total_bandwidth_saved_gbps: number
  active_training_runs: number
  total_optimizations: number
}

export interface Server {
  id: number
  hostname: string
  ip_address: string
  gpu_count: number
  gpu_model: string
  total_memory_gb: number
  status: string
  last_heartbeat: string | null
}

export interface GPU {
  index: number
  name: string
  memory_total_mb: number
  memory_used_mb: number
  utilization_pct: number
  temperature_c: number
}

export interface ServerDetail extends Server {
  gpus: GPU[]
}

export interface OptimizationSession {
  id: number
  session_id: string
  model_name: string
  model_type: string
  gpu_index: number
  speedup: number
  memory_saved_mb: number
  bandwidth_saved_gbps: number
  memory_bound_pct: number
  status: string
  optimizations_applied: number
  rollback_count: number
  started_at: string | null
  completed_at: string | null
}

export interface TrainingRun {
  id: number
  run_id: string
  model_name: string
  current_epoch: number
  total_epochs: number
  current_step: number
  total_steps: number
  current_loss: number | null
  baseline_loss: number | null
  speedup: number
  optimizations_applied: number
  rollbacks: number
  status: string
  started_at: string | null
  updated_at: string | null
}

export interface Alert {
  id: number
  level: string
  title: string
  message: string
  source: string
  acknowledged: boolean
  created_at: string | null
}

export interface MetricSnapshot {
  timestamp: string
  gpu_index: number
  memory_used_mb: number
  utilization_pct: number
  temperature_c: number
}

// API functions
export const api = {
  // Fleet
  getFleetOverview: () => fetchApi<FleetOverview>('/fleet/overview'),

  // Servers
  getServers: () => fetchApi<Server[]>('/servers'),
  getServer: (hostname: string) => fetchApi<ServerDetail>(`/servers/${hostname}`),
  getServerMetrics: (hostname: string, hours: number = 24) =>
    fetchApi<MetricSnapshot[]>(`/servers/${hostname}/metrics?hours=${hours}`),

  // Sessions
  getSessions: (params?: {
    limit?: number
    offset?: number
    server?: string
    model?: string
    status?: string
  }) => {
    const queryParams = new URLSearchParams()
    if (params?.limit) queryParams.set('limit', params.limit.toString())
    if (params?.offset) queryParams.set('offset', params.offset.toString())
    if (params?.server) queryParams.set('server', params.server)
    if (params?.model) queryParams.set('model', params.model)
    if (params?.status) queryParams.set('status', params.status)
    const query = queryParams.toString()
    return fetchApi<OptimizationSession[]>(`/sessions${query ? `?${query}` : ''}`)
  },
  getSession: (sessionId: string) => fetchApi<OptimizationSession>(`/sessions/${sessionId}`),

  // Training
  getTrainingRuns: (status?: string) => {
    const query = status ? `?status=${status}` : ''
    return fetchApi<TrainingRun[]>(`/training${query}`)
  },

  // Alerts
  getAlerts: (unacknowledgedOnly: boolean = false, limit: number = 50) =>
    fetchApi<Alert[]>(`/alerts?unacknowledged_only=${unacknowledgedOnly}&limit=${limit}`),
  acknowledgeAlert: (alertId: number) =>
    fetchApi<{ status: string }>(`/alerts/${alertId}/acknowledge`, { method: 'PATCH' }),

  // Export
  exportSessions: (format: 'json' | 'csv' = 'json') =>
    fetchApi<any>(`/export/sessions?format=${format}`),
}
