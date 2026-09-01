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
import { getStatusBadge, getStatusColor } from '@/lib/utils'

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
        <div className="text-gray-600">Loading system status...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">{error}</p>
      </div>
    )
  }

  const serviceStatus = health?.services || {}
  const modelInfo = models?.models || {}

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">System Overview</h1>
        <p className="text-gray-600">
          Real-time status of the AEGIS deepfake detection research system
        </p>
      </div>

      {/* System Health */}
      <div className="card">
        <h2 className="section-header">
          <Activity className="w-6 h-6" />
          System Health
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="metric-card">
            <div className="flex items-center justify-between mb-2">
              <span className="metric-label">API Status</span>
              <span className={`status-badge ${getStatusBadge(health?.status || 'unknown')}`}>
                {health?.status || 'Unknown'}
              </span>
            </div>
            <div className="metric-value text-green-600">
              {health?.status === 'healthy' ? 'Operational' : 'Degraded'}
            </div>
          </div>

          <div className="metric-card">
            <div className="flex items-center justify-between mb-2">
              <span className="metric-label">Version</span>
              <Cpu className="w-5 h-5 text-gray-400" />
            </div>
            <div className="metric-value">{health?.version || 'N/A'}</div>
          </div>

          <div className="metric-card">
            <div className="flex items-center justify-between mb-2">
              <span className="metric-label">Active Services</span>
              <Zap className="w-5 h-5 text-gray-400" />
            </div>
            <div className="metric-value">
              {Object.values(serviceStatus).filter(s => s === 'loaded').length} / {Object.keys(serviceStatus).length}
            </div>
          </div>
        </div>
      </div>

      {/* Model Status */}
      <div className="card">
        <h2 className="section-header">
          <HardDrive className="w-6 h-6" />
          Model Status
        </h2>

        <div className="space-y-4">
          {Object.entries(modelInfo).map(([modality, info]) => {
            const statusIcon = info.status === 'loaded' ? (
              <CheckCircle className="w-5 h-5 text-green-600" />
            ) : info.status.includes('not_found') ? (
              <AlertCircle className="w-5 h-5 text-yellow-600" />
            ) : (
              <XCircle className="w-5 h-5 text-red-600" />
            )

            return (
              <div
                key={modality}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-4">
                  {statusIcon}
                  <div>
                    <div className="font-medium text-gray-900 capitalize">{modality} Model</div>
                    <div className="text-sm text-gray-600">
                      {info.model_type} v{info.version}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`font-medium ${getStatusColor(info.status)}`}>
                    {info.status.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-gray-600">
                    {info.calibration_path ? 'Calibrated' : 'Uncalibrated'}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="metric-label">Supported Modalities</div>
          <div className="metric-value">3</div>
          <div className="text-sm text-gray-600">Image, Video, Audio</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Calibration</div>
          <div className="metric-value">
            {Object.values(modelInfo).filter(m => m.calibration_path).length}
          </div>
          <div className="text-sm text-gray-600">Models with temperature scaling</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Inference Type</div>
          <div className="metric-value">Real-time</div>
          <div className="text-sm text-gray-600">Batch processing available</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Research Mode</div>
          <div className="metric-value">Active</div>
          <div className="text-sm text-gray-600">Generalization analysis enabled</div>
        </div>
      </div>

      {/* System Information */}
      <div className="card">
        <h2 className="section-header">
          <Activity className="w-6 h-6" />
          System Information
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Research Focus</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• Cross-modal deepfake detection</li>
              <li>• Generalization gap analysis</li>
              <li>• Uncertainty calibration</li>
              <li>• Explainability and interpretability</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-3">Technical Stack</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• PyTorch deep learning framework</li>
              <li>• Temperature scaling calibration</li>
              <li>• Multi-modal architecture</li>
              <li>• FastAPI inference backend</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
