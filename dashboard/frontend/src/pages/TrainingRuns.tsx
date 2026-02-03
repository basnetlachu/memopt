import React, { useEffect, useState } from 'react'
import { Activity, Play, Pause, CheckCircle, XCircle, Zap, RotateCcw } from 'lucide-react'
import { api, TrainingRun } from '../utils/api'

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { color: string; icon: React.ElementType }> = {
    running: { color: 'bg-green-900 text-green-300', icon: Play },
    paused: { color: 'bg-yellow-900 text-yellow-300', icon: Pause },
    completed: { color: 'bg-blue-900 text-blue-300', icon: CheckCircle },
    failed: { color: 'bg-red-900 text-red-300', icon: XCircle },
  }

  const { color, icon: Icon } = config[status] || { color: 'bg-slate-700 text-slate-300', icon: Activity }

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs ${color}`}>
      <Icon className="w-3 h-3" />
      {status}
    </span>
  )
}

function ProgressBar({ current, total }: { current: number; total: number }) {
  const pct = total > 0 ? (current / total) * 100 : 0

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>Epoch {current}/{total}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-memopt-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function TrainingCard({ run }: { run: TrainingRun }) {
  const isActive = run.status === 'running'
  const lossChange = run.baseline_loss && run.current_loss
    ? ((run.current_loss - run.baseline_loss) / run.baseline_loss) * 100
    : null

  return (
    <div className={`card ${isActive ? 'border-memopt-500' : ''}`}>
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-medium text-white">{run.model_name}</h3>
          <p className="text-sm text-slate-400">Run: {run.run_id}</p>
        </div>
        <StatusBadge status={run.status} />
      </div>

      <div className="mb-4">
        <ProgressBar current={run.current_epoch} total={run.total_epochs} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <p className="text-slate-400">Current Loss</p>
          <p className="text-white font-medium">
            {run.current_loss?.toFixed(4) || 'N/A'}
          </p>
          {lossChange !== null && (
            <p className={`text-xs ${lossChange < 0 ? 'text-green-400' : 'text-red-400'}`}>
              {lossChange > 0 ? '+' : ''}{lossChange.toFixed(1)}%
            </p>
          )}
        </div>

        <div>
          <p className="text-slate-400">Speedup</p>
          <p className={`font-medium flex items-center gap-1 ${
            run.speedup > 1 ? 'text-green-400' : 'text-white'
          }`}>
            <Zap className="w-4 h-4" />
            {run.speedup.toFixed(2)}x
          </p>
        </div>

        <div>
          <p className="text-slate-400">Optimizations</p>
          <p className="text-white font-medium">{run.optimizations_applied}</p>
        </div>

        <div>
          <p className="text-slate-400">Rollbacks</p>
          <p className={`font-medium flex items-center gap-1 ${
            run.rollbacks > 0 ? 'text-yellow-400' : 'text-white'
          }`}>
            {run.rollbacks > 0 && <RotateCcw className="w-4 h-4" />}
            {run.rollbacks}
          </p>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-slate-700 flex justify-between text-xs text-slate-400">
        <span>
          Started: {run.started_at ? new Date(run.started_at).toLocaleString() : 'N/A'}
        </span>
        <span>
          Updated: {run.updated_at ? new Date(run.updated_at).toLocaleString() : 'N/A'}
        </span>
      </div>
    </div>
  )
}

export default function TrainingRuns() {
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [filter, setFilter] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await api.getTrainingRuns(filter || undefined)
        setRuns(data)
      } catch (error) {
        console.error('Failed to fetch training runs:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000) // Refresh every 5s
    return () => clearInterval(interval)
  }, [filter])

  const activeRuns = runs.filter((r) => r.status === 'running')
  const completedRuns = runs.filter((r) => r.status !== 'running')

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-memopt-500" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Training Runs</h1>
          <p className="text-slate-400">Active and recent training runs using Memopt</p>
        </div>

        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="input"
        >
          <option value="">All Status</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="stat-card">
          <p className="text-sm text-slate-400">Active Runs</p>
          <p className="text-3xl font-bold text-green-400">{activeRuns.length}</p>
        </div>
        <div className="stat-card">
          <p className="text-sm text-slate-400">Avg. Speedup</p>
          <p className="text-3xl font-bold text-memopt-400">
            {runs.length > 0
              ? (runs.reduce((sum, r) => sum + r.speedup, 0) / runs.length).toFixed(2)
              : '1.00'}x
          </p>
        </div>
        <div className="stat-card">
          <p className="text-sm text-slate-400">Total Optimizations</p>
          <p className="text-3xl font-bold text-white">
            {runs.reduce((sum, r) => sum + r.optimizations_applied, 0)}
          </p>
        </div>
      </div>

      {/* Active Runs */}
      {activeRuns.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            Active Runs
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {activeRuns.map((run) => (
              <TrainingCard key={run.id} run={run} />
            ))}
          </div>
        </div>
      )}

      {/* Completed Runs */}
      {completedRuns.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">Recent Runs</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {completedRuns.map((run) => (
              <TrainingCard key={run.id} run={run} />
            ))}
          </div>
        </div>
      )}

      {runs.length === 0 && (
        <div className="text-center py-12">
          <Activity className="w-16 h-16 mx-auto mb-4 text-slate-600" />
          <p className="text-lg text-slate-400">No training runs yet</p>
          <p className="text-sm text-slate-500 mt-2">
            Use <code className="bg-slate-700 px-2 py-1 rounded">@optimize_training()</code> decorator to track training
          </p>
        </div>
      )}
    </div>
  )
}
