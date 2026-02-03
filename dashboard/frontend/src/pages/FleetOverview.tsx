import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Server,
  Cpu,
  Zap,
  Activity,
  TrendingUp,
  HardDrive,
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
import { api, FleetOverview as FleetOverviewType, Server as ServerType, OptimizationSession } from '../utils/api'

function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'memopt'
}: {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ElementType
  color?: string
}) {
  const colorClasses: Record<string, string> = {
    memopt: 'from-memopt-600 to-memopt-700',
    green: 'from-green-600 to-green-700',
    purple: 'from-purple-600 to-purple-700',
    orange: 'from-orange-600 to-orange-700',
  }

  return (
    <div className="stat-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400">{title}</p>
          <p className="text-3xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-lg bg-gradient-to-br ${colorClasses[color]}`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
    </div>
  )
}

function ServerCard({ server }: { server: ServerType }) {
  const isOnline = server.status === 'online'

  return (
    <Link
      to={`/servers/${server.hostname}`}
      className="card hover:border-memopt-500 transition-colors"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Server className="w-5 h-5 text-memopt-500" />
          <span className="font-medium text-white">{server.hostname}</span>
        </div>
        <div className={`px-2 py-1 rounded text-xs ${
          isOnline ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
        }`}>
          {server.status}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <p className="text-slate-400">GPUs</p>
          <p className="text-white">{server.gpu_count}</p>
        </div>
        <div>
          <p className="text-slate-400">Model</p>
          <p className="text-white truncate">{server.gpu_model || 'N/A'}</p>
        </div>
        <div>
          <p className="text-slate-400">Memory</p>
          <p className="text-white">{server.total_memory_gb?.toFixed(0) || 0} GB</p>
        </div>
      </div>
    </Link>
  )
}

function RecentOptimization({ session }: { session: OptimizationSession }) {
  const speedupColor = session.speedup >= 1.2 ? 'text-green-400' :
                       session.speedup >= 1.1 ? 'text-yellow-400' : 'text-slate-400'

  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-700 last:border-0">
      <div className="flex-1">
        <p className="text-sm text-white">{session.model_name}</p>
        <p className="text-xs text-slate-400">
          {session.started_at ? new Date(session.started_at).toLocaleString() : 'N/A'}
        </p>
      </div>
      <div className="text-right">
        <p className={`text-sm font-medium ${speedupColor}`}>
          {session.speedup.toFixed(2)}x
        </p>
        <p className="text-xs text-slate-400">{session.status}</p>
      </div>
    </div>
  )
}

export default function FleetOverview() {
  const [overview, setOverview] = useState<FleetOverviewType | null>(null)
  const [servers, setServers] = useState<ServerType[]>([])
  const [recentSessions, setRecentSessions] = useState<OptimizationSession[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const [overviewData, serversData, sessionsData] = await Promise.all([
          api.getFleetOverview(),
          api.getServers(),
          api.getSessions({ limit: 10 }),
        ])
        setOverview(overviewData)
        setServers(serversData)
        setRecentSessions(sessionsData)
      } catch (error) {
        console.error('Failed to fetch fleet overview:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-memopt-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Fleet Overview</h1>
        <p className="text-slate-400">Real-time GPU optimization activity across all servers</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Servers"
          value={overview?.total_servers || 0}
          subtitle={`${overview?.online_servers || 0} online`}
          icon={Server}
          color="memopt"
        />
        <StatCard
          title="Total GPUs"
          value={overview?.total_gpus || 0}
          subtitle={`${overview?.total_memory_gb?.toFixed(0) || 0} GB total`}
          icon={Cpu}
          color="purple"
        />
        <StatCard
          title="Avg. Speedup"
          value={`${overview?.total_speedup?.toFixed(2) || '1.00'}x`}
          subtitle={`${overview?.total_optimizations || 0} optimizations`}
          icon={Zap}
          color="green"
        />
        <StatCard
          title="Active Training"
          value={overview?.active_training_runs || 0}
          subtitle="running now"
          icon={Activity}
          color="orange"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Server List */}
        <div className="lg:col-span-2">
          <div className="card">
            <h2 className="text-lg font-semibold text-white mb-4">Servers</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {servers.length > 0 ? (
                servers.map((server) => (
                  <ServerCard key={server.id} server={server} />
                ))
              ) : (
                <div className="col-span-2 text-center py-8 text-slate-400">
                  <Server className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>No servers registered yet</p>
                  <p className="text-sm mt-1">Run memopt daemon on your GPU servers</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Recent Optimizations */}
        <div className="card">
          <h2 className="text-lg font-semibold text-white mb-4">Recent Optimizations</h2>
          {recentSessions.length > 0 ? (
            <div className="space-y-1">
              {recentSessions.map((session) => (
                <RecentOptimization key={session.id} session={session} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-400">
              <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No optimizations yet</p>
            </div>
          )}

          <Link
            to="/sessions"
            className="block text-center text-memopt-400 hover:text-memopt-300 text-sm mt-4"
          >
            View all sessions →
          </Link>
        </div>
      </div>

      {/* Bandwidth Saved Chart (placeholder) */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-4">Bandwidth Saved (24h)</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={[
                { time: '00:00', saved: 0 },
                { time: '04:00', saved: 2.3 },
                { time: '08:00', saved: 5.1 },
                { time: '12:00', saved: 8.4 },
                { time: '16:00', saved: 12.2 },
                { time: '20:00', saved: 15.7 },
                { time: 'Now', saved: overview?.total_bandwidth_saved_gbps || 18.1 },
              ]}
              margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="colorSaved" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
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
                dataKey="saved"
                stroke="#0ea5e9"
                fillOpacity={1}
                fill="url(#colorSaved)"
                name="Bandwidth Saved (GB/s)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
