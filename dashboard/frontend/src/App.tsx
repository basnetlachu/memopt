import { useState, useEffect } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Server,
  Activity,
  History,
  Bell,
  Sun,
  Moon
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

function MemoptLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 769.38 535.93"
      className={className}
      fill="currentColor"
    >
      <path d="M2.97,26.3v26.3h150.26l19.73-22.36c10.52-12.17,20.71-24,22.03-26.3,1.97-2.96-18.08-3.95-94.69-3.95H2.97v26.3Z"/>
      <path d="M209.78,44.39l-40.11,44.39-84.17.66-84.17.99v59.18l82.53.99,82.2.66,42.74-39.78,42.41-39.78,54.25.99c29.92.33,146.31.33,259.09,0l204.51-.33V0H250.22l-40.44,44.39Z"/>
      <path d="M221.62,149.6l-39.45,37.81-90.42.66-90.42.99v59.18h19.73c10.85.33,57.54.66,103.57.99l84.17.66,32.55,26.96,32.88,27.29h493.18l.99-38.47.99-38.8-252.51-.66-252.51-.99v-36.17h503.05l.99-38.8.99-38.47-254.15.33h-254.15l-39.45,37.48Z"/>
      <path d="M1,290.32c-.99,2.63-1.32,16.11-.66,30.58l.99,25.97,102.58.99,102.91.66,38.14,36.17,37.81,36.17,242.32-.66,242.32-.99.99-38.8.99-38.47h-505.68l-35.51-27.95-35.51-27.95h-95.02c-76.28,0-95.35.99-96.66,4.27Z"/>
      <path d="M1.33,388.3c-1.32,1.64-1.32,15.12,0,29.59l2.3,25.97h179.85l42.09,46.03,42.09,46.03,249.88-.99,249.88-.66v-75.62l-250.54-1.64-250.54-1.64-38.14-35.51-38.47-35.18h-93.05c-62.47,0-94.03,1.32-95.35,3.62Z"/>
      <path d="M2.31,508.64c.66,14.14,1.97,25.97,2.3,26.63.33.33,47.35.33,104.56,0l103.57-.99-21.7-25.65-22.03-25.32H.67l1.64,25.32Z"/>
    </svg>
  )
}

function ThemeToggle({ darkMode, onToggle }: { darkMode: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className={`p-2 rounded-lg transition-colors ${
        darkMode
          ? 'bg-slate-700 hover:bg-slate-600 text-yellow-400'
          : 'bg-slate-200 hover:bg-slate-300 text-slate-700'
      }`}
      title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
    </button>
  )
}

function Sidebar({ darkMode, onToggleTheme }: { darkMode: boolean; onToggleTheme: () => void }) {
  const location = useLocation()

  return (
    <aside className={`w-64 border-r min-h-screen p-4 ${
      darkMode
        ? 'bg-slate-900 border-slate-700'
        : 'bg-white border-slate-200'
    }`}>
      <div className="flex items-center justify-between mb-8 px-2">
        <div className="flex items-center gap-3">
          <MemoptLogo className={`w-10 h-8 ${darkMode ? 'text-white' : 'text-black'}`} />
          <div>
            <h1 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-black'}`}>Memopt</h1>
            <p className={`text-xs ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>GPU Optimization</p>
          </div>
        </div>
        <ThemeToggle darkMode={darkMode} onToggle={onToggleTheme} />
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
                  : darkMode
                    ? 'text-slate-400 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-black'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>

      <div className="mt-8 px-2">
        <div className={`p-3 rounded-lg ${darkMode ? 'bg-slate-800' : 'bg-slate-100'}`}>
          <p className={`text-xs mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>Version</p>
          <p className={`text-sm ${darkMode ? 'text-white' : 'text-black'}`}>Memopt v0.4.0</p>
        </div>
      </div>
    </aside>
  )
}

function App() {
  const { connected } = useWebSocket()
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('memopt-dark-mode')
    return saved !== null ? JSON.parse(saved) : true
  })

  useEffect(() => {
    localStorage.setItem('memopt-dark-mode', JSON.stringify(darkMode))
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  const toggleTheme = () => setDarkMode(!darkMode)

  return (
    <div className={`flex min-h-screen ${darkMode ? 'bg-slate-950' : 'bg-slate-50'}`}>
      <Sidebar darkMode={darkMode} onToggleTheme={toggleTheme} />

      <main className={`flex-1 p-6 overflow-auto ${darkMode ? 'text-white' : 'text-black'}`}>
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className={`text-xs ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
              {connected ? 'Live updates enabled' : 'Connecting...'}
            </span>
          </div>
        </div>

        <div className={darkMode ? '' : 'light-mode'}>
          <Routes>
            <Route path="/" element={<FleetOverview />} />
            <Route path="/servers" element={<ServerDetail />} />
            <Route path="/servers/:hostname" element={<ServerDetail />} />
            <Route path="/training" element={<TrainingRuns />} />
            <Route path="/sessions" element={<SessionHistory />} />
            <Route path="/alerts" element={<Alerts />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

export default App
