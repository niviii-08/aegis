import { useState, useEffect } from 'react'
import { Eye, EyeOff, Lightbulb, GitBranch } from 'lucide-react'

export default function Explainability() {
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading explainability data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Explainability</h1>
        <p className="text-gray-600">
          Model interpretability and explanation methods for deepfake detection
        </p>
      </div>

      {/* Overview */}
      <div className="card">
        <h2 className="section-header">
          <Eye className="w-6 h-6" />
          Explainability Overview
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="metric-card">
            <div className="metric-label">Image Method</div>
            <div className="metric-value">Grad-CAM</div>
            <div className="text-sm text-gray-600">Spatial saliency maps</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Video Method</div>
            <div className="metric-value">Temporal Grad-CAM</div>
            <div className="text-sm text-gray-600">Frame-level attention</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Audio Method</div>
            <div className="metric-value">Saliency Maps</div>
            <div className="text-sm text-gray-600">Spectral attention</div>
          </div>
        </div>
      </div>

      {/* Image Explainability */}
      <div className="card">
        <h2 className="section-header">
          <Eye className="w-6 h-6" />
          Image Saliency Analysis
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Grad-CAM Implementation</h3>
            <p className="text-sm text-gray-600 mb-3">
              Gradient-weighted Class Activation Mapping highlights regions of the image
              that contribute most to the deepfake prediction.
            </p>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Uses final convolutional layer features</li>
              <li>• Computes gradient weights for target class</li>
              <li>• Generates spatial attention maps</li>
              <li>• Identifies manipulation artifacts</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-3">Analysis Capabilities</h3>
            <div className="space-y-2">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Face Region Focus</div>
                <div className="text-xs text-gray-600">Attention concentrated on facial features</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Artifact Detection</div>
                <div className="text-xs text-gray-600">Highlights boundary inconsistencies</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Spatial Frequency</div>
                <div className="text-xs text-gray-600">Detects frequency domain artifacts</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Video Explainability */}
      <div className="card">
        <h2 className="section-header">
          <EyeOff className="w-6 h-6" />
          Video Temporal Analysis
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Temporal Grad-CAM</h3>
            <p className="text-sm text-gray-600 mb-3">
              Extended Grad-CAM for video sequences that analyzes both spatial and temporal
              dimensions to identify frame-level inconsistencies.
            </p>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Analyzes frame sequences</li>
              <li>• Identifies temporal inconsistencies</li>
              <li>• Tracks attention across frames</li>
              <li>• Detects flickering artifacts</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-3">Frame-Level Analysis</h3>
            <div className="space-y-2">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Consistency Tracking</div>
                <div className="text-xs text-gray-600">Monitors feature stability across frames</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Temporal Artifacts</div>
                <div className="text-xs text-gray-600">Detects unnatural motion patterns</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Sequence Attention</div>
                <div className="text-xs text-gray-600">Highlights important temporal segments</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Audio Explainability */}
      <div className="card">
        <h2 className="section-header">
          <Lightbulb className="w-6 h-6" />
          Audio Spectral Analysis
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Spectral Saliency</h3>
            <p className="text-sm text-gray-600 mb-3">
              Audio saliency maps identify which frequency bands and temporal regions
              contribute most to the deepfake detection decision.
            </p>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Mel-spectrogram attention</li>
              <li>• Frequency band importance</li>
              <li>• Temporal segment highlighting</li>
              <li>• Artifact localization</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-3">Analysis Features</h3>
            <div className="space-y-2">
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Frequency Attention</div>
                <div className="text-xs text-gray-600">Identifies suspicious frequency ranges</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Temporal Segments</div>
                <div className="text-xs text-gray-600">Highlights suspicious time regions</div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-900">Spectral Patterns</div>
                <div className="text-xs text-gray-600">Detects unnatural spectral characteristics</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Integration Status */}
      <div className="card">
        <h2 className="section-header">
          <GitBranch className="w-6 h-6" />
          Integration Status
        </h2>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">Dashboard Integration</div>
              <div className="text-sm text-gray-600">Real-time explanation visualization</div>
            </div>
            <span className="status-badge status-warning">In Progress</span>
          </div>

          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">API Response Integration</div>
              <div className="text-sm text-gray-600">Explanation metadata in prediction responses</div>
            </div>
            <span className="status-badge status-healthy">Complete</span>
          </div>

          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">Interactive Visualization</div>
              <div className="text-sm text-gray-600">Dynamic saliency map overlays</div>
            </div>
            <span className="status-badge status-warning">Planned</span>
          </div>
        </div>
      </div>
    </div>
  )
}
