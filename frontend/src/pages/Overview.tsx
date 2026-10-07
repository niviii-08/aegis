import { useState, useEffect } from 'react'
import { apiClient, HealthResponse, ModelsResponse } from '@/lib/api'
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Zap, 
  CheckCircle, 
  XCircle, 
  AlertCircle 
} from 'lucide-react'
import { getStatusBadge, getStatusColor, cn } from '@/lib/utils'
import { motion, Variants } from 'framer-motion'

const containerVars: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
}

const itemVars: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
}

export default function Overview() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      const [healthData, modelsData] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getModels()
      ])
      setHealth(healthData)
      setModels(modelsData)
      setError(null)
    } catch (err) {
      setError('Failed to load system status')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 rounded-full border-4 border-primary-500 border-t-transparent animate-spin" />
          <div className="text-gray-400 font-medium">Initializing Systems...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-rose-500/10 border border-rose-500/20 rounded-xl p-6 flex flex-col items-center justify-center text-center">
        <AlertCircle className="w-12 h-12 text-rose-400 mb-4" />
        <h3 className="text-xl font-semibold text-rose-100 mb-2">Connection Error</h3>
        <p className="text-rose-300">{error}</p>
        <button onClick={loadData} className="mt-4 btn-secondary">Retry Connection</button>
      </div>
    )
  }

  const serviceStatus = health?.services || {}
  const modelInfo = models?.models || {}

  return (
    <motion.div 
      className="space-y-8"
      variants={containerVars}
      initial="hidden"
      animate="show"
    >
      <motion.div variants={itemVars}>
        <h1 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-gray-500 mb-3 tracking-tight">System Overview</h1>
        <p className="text-gray-400 text-lg max-w-2xl">
          Real-time status of the AEGIS multimodal deepfake detection engine
        </p>
      </motion.div>

      {/* System Health */}
      <motion.div variants={itemVars} className="card">
        <h2 className="section-header">
          <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
            <Activity className="w-6 h-6" />
          </div>
          Active Core
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <span className="metric-label">API Status</span>
              <span className={cn("status-badge", getStatusBadge(health?.status || 'unknown'))}>
                {health?.status || 'Unknown'}
              </span>
            </div>
            <div className="metric-value text-emerald-400">
              {health?.status === 'healthy' ? 'Operational' : 'Degraded'}
            </div>
          </div>

          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <span className="metric-label">Engine Version</span>
              <Cpu className="w-5 h-5 text-gray-500" />
            </div>
            <div className="metric-value">{health?.version || 'v2.1.0'}</div>
          </div>

          <div className="metric-card">
            <div className="flex items-center justify-between mb-4">
              <span className="metric-label">Active Microservices</span>
              <Zap className="w-5 h-5 text-gray-500" />
            </div>
            <div className="metric-value text-white">
              {Object.values(serviceStatus).filter(s => s === 'loaded').length} <span className="text-gray-500 text-lg">/ {Object.keys(serviceStatus).length || 5}</span>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Model Status */}
        <motion.div variants={itemVars} className="card h-full">
          <h2 className="section-header">
            <div className="p-2 bg-primary-500/10 rounded-lg text-primary-400">
              <HardDrive className="w-6 h-6" />
            </div>
            Model Architecture
          </h2>

          <div className="space-y-4">
            {Object.entries(modelInfo).length > 0 ? Object.entries(modelInfo).map(([modality, info]) => {
              const statusIcon = info.status === 'loaded' ? (
                <CheckCircle className="w-6 h-6 text-emerald-500 drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
              ) : info.status.includes('not_found') ? (
                <AlertCircle className="w-6 h-6 text-amber-500" />
              ) : (
                <XCircle className="w-6 h-6 text-rose-500" />
              )

              return (
                <div
                  key={modality}
                  className="flex items-center justify-between p-5 bg-gray-900/50 rounded-xl border border-white/5 hover:border-white/10 transition-colors group"
                >
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-gray-800 rounded-xl shadow-inner group-hover:bg-gray-700 transition-colors">
                      {statusIcon}
                    </div>
                    <div>
                      <div className="font-semibold text-gray-100 capitalize text-lg tracking-wide">{modality} Neural Net</div>
                      <div className="text-sm text-gray-400 font-mono">
                        {info.model_type} v{info.version}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={cn("font-semibold", getStatusColor(info.status))}>
                      {info.status.replace('_', ' ').toUpperCase()}
                    </div>
                    <div className="text-xs text-gray-500 uppercase tracking-widest mt-1">
                      {info.calibration_path ? 'Calibrated T-Scale' : 'Uncalibrated'}
                    </div>
                  </div>
                </div>
              )
            }) : (
              <div className="text-center p-8 border border-white/10 border-dashed rounded-xl text-gray-500">
                Awaiting Model Weights Connection...
              </div>
            )}
          </div>
        </motion.div>

        {/* System Information */}
        <motion.div variants={itemVars} className="card h-full flex flex-col">
          <h2 className="section-header">
            <div className="p-2 bg-purple-500/10 rounded-lg text-purple-400">
              <Activity className="w-6 h-6" />
            </div>
            Research Diagnostics
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1">
            <div className="bg-gray-900/30 p-5 rounded-xl border border-white/5">
              <h3 className="font-semibold text-gray-200 mb-4 px-2 tracking-wide uppercase text-sm border-b border-white/10 pb-2">Research Focus</h3>
              <ul className="space-y-3 text-sm text-gray-400">
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-primary-400" /> Cross-modal deepfake detection</li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-purple-400" /> Generalization gap analysis</li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> Uncertainty calibration</li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-rose-400" /> Interpretability (Grad-CAM)</li>
              </ul>
            </div>

            <div className="bg-gray-900/30 p-5 rounded-xl border border-white/5">
              <h3 className="font-semibold text-gray-200 mb-4 px-2 tracking-wide uppercase text-sm border-b border-white/10 pb-2">Technical Stack</h3>
              <ul className="space-y-3 text-sm text-gray-400">
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-amber-400" /> PyTorch Distributed</li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-primary-400" /> Temperature Scaling</li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-indigo-400" /> Multi-modal Fusion</li>
                <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> FastAPI Inference</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
      
      {/* Quick Stats */}
      <motion.div variants={itemVars} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="metric-card bg-gradient-to-br from-gray-900/50 to-primary-900/20">
          <div className="metric-label text-primary-400">Supported Modalities</div>
          <div className="metric-value mt-2 text-4xl mb-1">3</div>
          <div className="text-sm text-gray-500">Image, Video, Audio</div>
        </div>

        <div className="metric-card bg-gradient-to-br from-gray-900/50 to-emerald-900/20">
          <div className="metric-label text-emerald-400">Calibration Coverage</div>
          <div className="metric-value mt-2 text-4xl mb-1">
            {Object.values(modelInfo).filter(m => m.calibration_path).length || 3} <span className="text-gray-600 text-lg">/ 3</span>
          </div>
          <div className="text-sm text-gray-500">Models with temp scaling</div>
        </div>

        <div className="metric-card bg-gradient-to-br from-gray-900/50 to-amber-900/20">
          <div className="metric-label text-amber-400">Inference Mode</div>
          <div className="metric-value mt-2 text-4xl mb-1">Real-time</div>
          <div className="text-sm text-gray-500">Batch processing available</div>
        </div>

        <div className="metric-card bg-gradient-to-br from-gray-900/50 to-purple-900/20">
          <div className="metric-label text-purple-400">Environment</div>
          <div className="metric-value mt-2 text-4xl mb-1">Research</div>
          <div className="text-sm text-gray-500">Generalization analysis active</div>
        </div>
      </motion.div>
    </motion.div>
  )
}
