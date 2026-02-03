import React, { useEffect, useState } from 'react'
import { History, Search, Download, Filter, Zap, RotateCcw, CheckCircle, XCircle, Clock } from 'lucide-react'
import { api, OptimizationSession } from '../utils/api'

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle className="w-4 h-4 text-green-400" />
    case 'failed':
      return <XCircle className="w-4 h-4 text-red-400" />
    case 'rolled_back':
      return <RotateCcw className="w-4 h-4 text-yellow-400" />
    case 'profiling':
    case 'optimizing':
      return <Clock className="w-4 h-4 text-blue-400 animate-pulse" />
    default:
      return <Clock className="w-4 h-4 text-slate-400" />
  }
}

function SessionRow({ session }: { session: OptimizationSession }) {
  const speedupColor = session.speedup >= 1.2 ? 'text-green-400' :
                       session.speedup >= 1.1 ? 'text-yellow-400' :
                       session.speedup > 1.0 ? 'text-blue-400' : 'text-slate-400'

  return (
    <tr className="border-b border-slate-700 hover:bg-slate-800/50">
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <StatusIcon status={session.status} />
          <span className="text-slate-400 font-mono text-sm">{session.session_id}</span>
        </div>
      </td>
      <td className="py-3 px-4">
        <div>
          <p className="text-white">{session.model_name}</p>
          <p className="text-xs text-slate-400">{session.model_type}</p>
        </div>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="text-slate-300">GPU {session.gpu_index}</span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className={`font-medium flex items-center justify-center gap-1 ${speedupColor}`}>
          <Zap className="w-4 h-4" />
          {session.speedup.toFixed(2)}x
        </span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="text-slate-300">{session.memory_saved_mb.toFixed(0)} MB</span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="text-slate-300">{session.memory_bound_pct.toFixed(0)}%</span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className={`px-2 py-1 rounded text-xs ${
          session.status === 'completed' ? 'bg-green-900 text-green-300' :
          session.status === 'failed' ? 'bg-red-900 text-red-300' :
          session.status === 'rolled_back' ? 'bg-yellow-900 text-yellow-300' :
          'bg-blue-900 text-blue-300'
        }`}>
          {session.status}
        </span>
      </td>
      <td className="py-3 px-4 text-right">
        <span className="text-slate-400 text-sm">
          {session.started_at ? new Date(session.started_at).toLocaleString() : 'N/A'}
        </span>
      </td>
    </tr>
  )
}

export default function SessionHistory() {
  const [sessions, setSessions] = useState<OptimizationSession[]>([])
  const [loading, setLoading] = useState(true)
  const [searchModel, setSearchModel] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [page, setPage] = useState(0)
  const pageSize = 25

  useEffect(() => {
    async function fetchData() {
      setLoading(true)
      try {
        const data = await api.getSessions({
          limit: pageSize,
          offset: page * pageSize,
          model: searchModel || undefined,
          status: filterStatus || undefined,
        })
        setSessions(data)
      } catch (error) {
        console.error('Failed to fetch sessions:', error)
      } finally {
        setLoading(false)
      }
    }

    const debounce = setTimeout(fetchData, 300)
    return () => clearTimeout(debounce)
  }, [page, searchModel, filterStatus])

  const handleExport = async (format: 'json' | 'csv') => {
    try {
      const data = await api.exportSessions(format)
      const blob = new Blob([format === 'json' ? JSON.stringify(data, null, 2) : data], {
        type: format === 'json' ? 'application/json' : 'text/csv',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `memopt_sessions.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  // Summary stats
  const completedSessions = sessions.filter((s) => s.status === 'completed')
  const avgSpeedup = completedSessions.length > 0
    ? completedSessions.reduce((sum, s) => sum + s.speedup, 0) / completedSessions.length
    : 1.0
  const totalMemorySaved = sessions.reduce((sum, s) => sum + s.memory_saved_mb, 0)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Session History</h1>
          <p className="text-slate-400">All optimization sessions</p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => handleExport('csv')}
            className="btn-primary flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="stat-card">
          <p className="text-sm text-slate-400">Total Sessions</p>
          <p className="text-3xl font-bold text-white">{sessions.length}</p>
        </div>
        <div className="stat-card">
          <p className="text-sm text-slate-400">Avg. Speedup</p>
          <p className="text-3xl font-bold text-green-400">{avgSpeedup.toFixed(2)}x</p>
        </div>
        <div className="stat-card">
          <p className="text-sm text-slate-400">Memory Saved</p>
          <p className="text-3xl font-bold text-memopt-400">{(totalMemorySaved / 1024).toFixed(1)} GB</p>
        </div>
        <div className="stat-card">
          <p className="text-sm text-slate-400">Success Rate</p>
          <p className="text-3xl font-bold text-white">
            {sessions.length > 0
              ? ((completedSessions.length / sessions.length) * 100).toFixed(0)
              : 0}%
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search by model name..."
            value={searchModel}
            onChange={(e) => {
              setSearchModel(e.target.value)
              setPage(0)
            }}
            className="input w-full pl-10"
          />
        </div>

        <select
          value={filterStatus}
          onChange={(e) => {
            setFilterStatus(e.target.value)
            setPage(0)
          }}
          className="input"
        >
          <option value="">All Status</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="rolled_back">Rolled Back</option>
          <option value="profiling">Profiling</option>
        </select>
      </div>

      {/* Table */}
      <div className="card overflow-hidden p-0">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-memopt-500" />
          </div>
        ) : sessions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-slate-800/50 text-left">
                  <th className="py-3 px-4 text-sm font-medium text-slate-400">Session ID</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400">Model</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400 text-center">GPU</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400 text-center">Speedup</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400 text-center">Memory Saved</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400 text-center">Memory Bound</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400 text-center">Status</th>
                  <th className="py-3 px-4 text-sm font-medium text-slate-400 text-right">Date</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((session) => (
                  <SessionRow key={session.id} session={session} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <History className="w-16 h-16 mx-auto mb-4 text-slate-600" />
            <p className="text-lg text-slate-400">No sessions found</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {sessions.length >= pageSize && (
        <div className="flex items-center justify-between">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="btn-primary disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-slate-400">Page {page + 1}</span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={sessions.length < pageSize}
            className="btn-primary disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
