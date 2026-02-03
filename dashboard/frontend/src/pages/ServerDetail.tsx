import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  Server,
  Cpu,
  Thermometer,
  HardDrive,
  Activity,
  ArrowLeft,
  Clock
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts'
import { api, Server as ServerType, ServerDetail as ServerDetailType, MetricSnapshot } from '../utils/api'

function GPUCard({ gpu, index }: { gpu: any; index: number }) {
  const memoryPct = (gpu.memory_used_mb / gpu.memory_total_mb) * 100
  const memoryColor = memoryPct > 90 ? 'bg-red-500' : memoryPct > 70 ? 'bg-yellow-500' : 'bg-green-500'
  const utilColor = gpu.utilization_pct > 90 ? 'text-green-400' : gpu.utilization_pct > 50 ? 'text-yellow-400' : 'text-slate-400'

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-memopt-500" />
          <span className="font-medium text-white">GPU {index}</span>
        </div>
        <span className="text-sm text-slate-400">{gpu.name}</span>
      </div>

      <div className="space-y-4">
        {/* Memory */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-400">Memory</span>
            <span className="text-white">
              {(gpu.memory_used_mb / 1024).toFixed(1)} / {(gpu.memory_total_mb / 1024).toFixed(1)} GB
            </span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className={`h-full ${memoryColor} transition-all duration-300`}
              style={{ width: `${memoryPct}%` }}
            />
          </div>
        </div>

        {/* Utilization */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-slate-400" />
            <span className="text-slate-400">Utilization</span>
          </div>
          <span className={`font-medium ${utilColor}`}>
            {gpu.utilization_pct.toFixed(0)}%
          </span>
        </div>

        {/* Temperature */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Thermometer className="w-4 h-4 text-slate-400" />
            <span className="text-slate-400">Temperature</span>
          </div>
          <span className={`font-medium ${
            gpu.temperature_c > 80 ? 'text-red-400' :
            gpu.temperature_c > 60 ? 'text-yellow-400' : 'text-green-400'
          }`}>
            {gpu.temperature_c}°C
          </span>
        </div>
      </div>
    </div>
  )
}

function ServerList({ servers, selectedHostname }: { servers: ServerType[]; selectedHostname?: string }) {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-white mb-4">All Servers</h2>
      <div className="space-y-2">
        {servers.map((server) => (
          <Link
            key={server.id}
            to={`/servers/${server.hostname}`}
            className={`block p-3 rounded-lg transition-colors ${
              server.hostname === selectedHostname
                ? 'bg-memopt-600'
                : 'hover:bg-slate-700'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  server.status === 'online' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-white">{server.hostname}</span>
              </div>
              <span className="text-sm text-slate-400">{server.gpu_count} GPUs</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

export default function ServerDetail() {
  const { hostname } = useParams<{ hostname: string }>()
  const [servers, setServers] = useState<ServerType[]>([])
  const [server, setServer] = useState<ServerDetailType | null>(null)
  const [metrics, setMetrics] = useState<MetricSnapshot[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchServers() {
      try {
        const serversData = await api.getServers()
        setServers(serversData)
      } catch (error) {
        console.error('Failed to fetch servers:', error)
      }
    }
    fetchServers()
  }, [])

  useEffect(() => {
    if (!hostname) {
      setLoading(false)
      return
    }

    async function fetchServerData() {
      setLoading(true)
      try {
        const [serverData, metricsData] = await Promise.all([
          api.getServer(hostname),
          api.getServerMetrics(hostname, 24),
        ])
        setServer(serverData)
        setMetrics(metricsData)
      } catch (error) {
        console.error('Failed to fetch server data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchServerData()
    const interval = setInterval(fetchServerData, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [hostname])

  // If no hostname, show server list
  if (!hostname) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Servers</h1>
          <p className="text-slate-400">Select a server to view details</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {servers.map((server) => (
            <Link
              key={server.id}
              to={`/servers/${server.hostname}`}
              className="card hover:border-memopt-500 transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Server className="w-5 h-5 text-memopt-500" />
                  <span className="font-medium text-white">{server.hostname}</span>
                </div>
                <div className={`px-2 py-1 rounded text-xs ${
                  server.status === 'online' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                }`}>
                  {server.status}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-slate-400">GPUs</p>
                  <p className="text-white">{server.gpu_count}x {server.gpu_model}</p>
                </div>
                <div>
                  <p className="text-slate-400">Memory</p>
                  <p className="text-white">{server.total_memory_gb?.toFixed(0) || 0} GB</p>
                </div>
              </div>
            </Link>
          ))}

          {servers.length === 0 && (
            <div className="col-span-full text-center py-12 text-slate-400">
              <Server className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No servers registered</p>
              <p className="text-sm mt-2">Run <code className="bg-slate-700 px-2 py-1 rounded">memopt daemon start</code> on your GPU servers</p>
            </div>
          )}
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-memopt-500" />
      </div>
    )
  }

  if (!server) {
    return (
      <div className="text-center py-12">
        <p className="text-slate-400">Server not found</p>
        <Link to="/servers" className="text-memopt-400 hover:text-memopt-300 mt-2 inline-block">
          ← Back to servers
        </Link>
      </div>
    )
  }

  // Prepare chart data
  const chartData = metrics.slice(-50).map((m) => ({
    time: new Date(m.timestamp).toLocaleTimeString(),
    utilization: m.utilization_pct,
    memory: (m.memory_used_mb / 1024).toFixed(1),
    temperature: m.temperature_c,
  }))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/servers" className="p-2 hover:bg-slate-800 rounded-lg">
          <ArrowLeft className="w-5 h-5 text-slate-400" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">{server.hostname}</h1>
          <p className="text-slate-400">{server.ip_address}</p>
        </div>
        <div className={`px-3 py-1 rounded-lg text-sm ${
          server.status === 'online' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
        }`}>
          {server.status}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar - Server List */}
        <div className="lg:col-span-1">
          <ServerList servers={servers} selectedHostname={hostname} />
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* GPU Cards */}
          <div>
            <h2 className="text-lg font-semibold text-white mb-4">GPUs</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {server.gpus.map((gpu, index) => (
                <GPUCard key={index} gpu={gpu} index={index} />
              ))}
            </div>
          </div>

          {/* Utilization Chart */}
          <div className="card">
            <h2 className="text-lg font-semibold text-white mb-4">GPU Utilization (24h)</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#64748b" />
                  <YAxis stroke="#64748b" domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="utilization"
                    stroke="#0ea5e9"
                    strokeWidth={2}
                    dot={false}
                    name="Utilization %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Memory Chart */}
          <div className="card">
            <h2 className="text-lg font-semibold text-white mb-4">Memory Usage (24h)</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorMemory" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                    labelStyle={{ color: '#e2e8f0' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="memory"
                    stroke="#8b5cf6"
                    fillOpacity={1}
                    fill="url(#colorMemory)"
                    name="Memory (GB)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
