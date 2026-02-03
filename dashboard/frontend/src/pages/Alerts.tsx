import React, { useEffect, useState } from 'react'
import {
  Bell,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  Info,
  Check,
  Trash2
} from 'lucide-react'
import { api, Alert } from '../utils/api'

function AlertIcon({ level }: { level: string }) {
  switch (level) {
    case 'success':
      return <CheckCircle className="w-5 h-5 text-green-400" />
    case 'warning':
      return <AlertTriangle className="w-5 h-5 text-yellow-400" />
    case 'error':
      return <AlertCircle className="w-5 h-5 text-red-400" />
    default:
      return <Info className="w-5 h-5 text-blue-400" />
  }
}

function AlertCard({
  alert,
  onAcknowledge,
}: {
  alert: Alert
  onAcknowledge: (id: number) => void
}) {
  const borderColor = {
    success: 'border-l-green-500',
    warning: 'border-l-yellow-500',
    error: 'border-l-red-500',
    info: 'border-l-blue-500',
  }[alert.level] || 'border-l-slate-500'

  return (
    <div
      className={`card border-l-4 ${borderColor} ${
        alert.acknowledged ? 'opacity-60' : ''
      }`}
    >
      <div className="flex items-start gap-4">
        <AlertIcon level={alert.level} />

        <div className="flex-1">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-medium text-white">{alert.title}</h3>
              <p className="text-sm text-slate-400 mt-1">{alert.message}</p>
            </div>

            {!alert.acknowledged && (
              <button
                onClick={() => onAcknowledge(alert.id)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                title="Acknowledge"
              >
                <Check className="w-4 h-4 text-slate-400" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-4 mt-3 text-xs text-slate-500">
            <span className="capitalize">{alert.source}</span>
            <span>•</span>
            <span>
              {alert.created_at
                ? new Date(alert.created_at).toLocaleString()
                : 'Unknown'}
            </span>
            {alert.acknowledged && (
              <>
                <span>•</span>
                <span className="text-green-500">Acknowledged</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [showAcknowledged, setShowAcknowledged] = useState(true)

  async function fetchAlerts() {
    try {
      const data = await api.getAlerts(!showAcknowledged, 100)
      setAlerts(data)
    } catch (error) {
      console.error('Failed to fetch alerts:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAlerts()
    const interval = setInterval(fetchAlerts, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [showAcknowledged])

  const handleAcknowledge = async (alertId: number) => {
    try {
      await api.acknowledgeAlert(alertId)
      setAlerts((prev) =>
        prev.map((a) => (a.id === alertId ? { ...a, acknowledged: true } : a))
      )
    } catch (error) {
      console.error('Failed to acknowledge alert:', error)
    }
  }

  const unacknowledgedCount = alerts.filter((a) => !a.acknowledged).length

  // Group alerts by level
  const errorAlerts = alerts.filter((a) => a.level === 'error')
  const warningAlerts = alerts.filter((a) => a.level === 'warning')
  const successAlerts = alerts.filter((a) => a.level === 'success')
  const infoAlerts = alerts.filter((a) => a.level === 'info')

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
          <h1 className="text-2xl font-bold text-white">Alerts</h1>
          <p className="text-slate-400">
            {unacknowledgedCount > 0
              ? `${unacknowledgedCount} unacknowledged alert${unacknowledgedCount !== 1 ? 's' : ''}`
              : 'All alerts acknowledged'}
          </p>
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showAcknowledged}
            onChange={(e) => setShowAcknowledged(e.target.checked)}
            className="rounded border-slate-600 bg-slate-700 text-memopt-500 focus:ring-memopt-500"
          />
          <span className="text-sm text-slate-400">Show acknowledged</span>
        </label>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="stat-card">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-slate-400">Errors</span>
          </div>
          <p className="text-2xl font-bold text-red-400 mt-2">{errorAlerts.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            <span className="text-sm text-slate-400">Warnings</span>
          </div>
          <p className="text-2xl font-bold text-yellow-400 mt-2">{warningAlerts.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">Success</span>
          </div>
          <p className="text-2xl font-bold text-green-400 mt-2">{successAlerts.length}</p>
        </div>
        <div className="stat-card">
          <div className="flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Info</span>
          </div>
          <p className="text-2xl font-bold text-blue-400 mt-2">{infoAlerts.length}</p>
        </div>
      </div>

      {/* Alerts List */}
      {alerts.length > 0 ? (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <AlertCard
              key={alert.id}
              alert={alert}
              onAcknowledge={handleAcknowledge}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <Bell className="w-16 h-16 mx-auto mb-4 text-slate-600" />
          <p className="text-lg text-slate-400">No alerts</p>
          <p className="text-sm text-slate-500 mt-2">
            Alerts will appear when optimizations are applied or issues are detected
          </p>
        </div>
      )}
    </div>
  )
}
