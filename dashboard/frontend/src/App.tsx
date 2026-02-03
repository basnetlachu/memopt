import React from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Server,
  Activity,
  History,
  Bell,
  Cpu
} from 'lucide-react'

import FleetOverview from './pages/FleetOverview'
import ServerDetail from './pages/ServerDetail'
import TrainingRuns from './pages/TrainingRuns'
import SessionHistory from './pages/SessionHistory'
import Alerts from './pages/Alerts'
import { useWebSocket } from './hooks/useWebSocket'

const navItems = [
  { path: '/', label: 'Fleet Overview', icon: LayoutDashboard },
  { path: '/servers', label: 'Servers', icon: Server },
  { path: '/training', label: 'Training Runs', icon: Activity },
  { path: '/sessions', label: 'Session History', icon: History },
  { path: '/alerts', label: 'Alerts', icon: Bell },
]

function Sidebar() {
  const location = useLocation()

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-700 min-h-screen p-4">
      <div className="flex items-center gap-3 mb-8 px-2">
        <Cpu className="w-8 h-8 text-memopt-500" />
        <div>
          <h1 className="text-xl font-bold text-white">Memopt</h1>
          <p className="text-xs text-slate-400">GPU Optimization Platform</p>
        </div>
      </div>

      <nav className="space-y-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path ||
            (item.path !== '/' && location.pathname.startsWith(item.path))
          const Icon = item.icon

          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                isActive
                  ? 'bg-memopt-600 text-white'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>

      <div className="mt-8 px-2">
        <div className="p-3 bg-slate-800 rounded-lg">
          <p className="text-xs text-slate-400 mb-1">Version</p>
          <p className="text-sm text-white">Memopt v0.4.0</p>
        </div>
      </div>
    </aside>
  )
}

function App() {
  const { connected, lastMessage } = useWebSocket()

  return (
    <div className="flex">
      <Sidebar />

      <main className="flex-1 p-6 overflow-auto">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-xs text-slate-400">
              {connected ? 'Live updates enabled' : 'Connecting...'}
            </span>
          </div>
        </div>

        <Routes>
          <Route path="/" element={<FleetOverview />} />
          <Route path="/servers" element={<ServerDetail />} />
          <Route path="/servers/:hostname" element={<ServerDetail />} />
          <Route path="/training" element={<TrainingRuns />} />
          <Route path="/sessions" element={<SessionHistory />} />
          <Route path="/alerts" element={<Alerts />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
