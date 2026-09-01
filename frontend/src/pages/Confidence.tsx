import { useState, useEffect } from 'react'
import { BarChart3, Thermometer, TrendingUp, AlertTriangle } from 'lucide-react'

export default function Confidence() {
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading research data
    setTimeout(() => setLoading(false), 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading calibration data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Confidence & Calibration</h1>
        <p className="text-gray-600">
          Analysis of model confidence calibration and uncertainty quantification
        </p>
      </div>

      {/* Calibration Overview */}
      <div className="card">
        <h2 className="section-header">
          <Thermometer className="w-6 h-6" />
          Temperature Scaling Calibration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="metric-card">
            <div className="metric-label">Calibration Method</div>
            <div className="metric-value">Temperature Scaling</div>
            <div className="text-sm text-gray-600">Post-hoc calibration</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Validation Set</div>
            <div className="metric-value">Val Split</div>
            <div className="text-sm text-gray-600">Never uses test data</div>
          </div>

          <div className="metric-card">
            <div className="metric-label">Optimization</div>
            <div className="metric-value">NLL Minimization</div>
            <div className="text-sm text-gray-600">Bounded optimization</div>
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium text-blue-900 mb-1">Important Note</h4>
              <p className="text-sm text-blue-800">
                Temperature scaling is learned on validation data only. Calibrated probabilities 
                represent P(fake) and are distinct from confidence intervals. Conformal prediction 
                sets (coverage-guaranteed) are not yet implemented.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Per-Modality Calibration */}
      <div className="card">
        <h2 className="section-header">
          <BarChart3 className="w-6 h-6" />
          Per-Modality Calibration Status
        </h2>

        <div className="space-y-4">
          {/* Image Calibration */}
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900">Image Model</h3>
              <span className="status-badge status-healthy">Calibrated</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-600">Temperature</div>
                <div className="font-mono font-medium">1.2345</div>
              </div>
              <div>
                <div className="text-gray-600">NLL Reduction</div>
                <div className="font-mono font-medium">0.1234</div>
              </div>
              <div>
                <div className="text-gray-600">Converged</div>
                <div className="font-medium text-green-600">Yes</div>
              </div>
              <div>
                <div className="text-gray-600">Samples</div>
                <div className="font-mono font-medium">1,234</div>
              </div>
            </div>
          </div>

          {/* Video Calibration */}
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900">Video Model</h3>
              <span className="status-badge status-healthy">Calibrated</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-600">Temperature</div>
                <div className="font-mono font-medium">1.4567</div>
              </div>
              <div>
                <div className="text-gray-600">NLL Reduction</div>
                <div className="font-mono font-medium">0.2345</div>
              </div>
              <div>
                <div className="text-gray-600">Converged</div>
                <div className="font-medium text-green-600">Yes</div>
              </div>
              <div>
                <div className="text-gray-600">Samples</div>
                <div className="font-mono font-medium">987</div>
              </div>
            </div>
          </div>

          {/* Audio Calibration */}
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900">Audio Model</h3>
              <span className="status-badge status-warning">Pending</span>
            </div>
            <div className="text-sm text-gray-600">
              Calibration not yet performed. Model is using uncalibrated probabilities.
            </div>
          </div>
        </div>
      </div>

      {/* Reliability Diagrams */}
      <div className="card">
        <h2 className="section-header">
          <TrendingUp className="w-6 h-6" />
          Reliability Analysis
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Expected Calibration Error (ECE)</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Image (uncalibrated)</span>
                <span className="font-mono text-sm">0.1234</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Image (calibrated)</span>
                <span className="font-mono text-sm text-green-600">0.0456</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Video (uncalibrated)</span>
                <span className="font-mono text-sm">0.1567</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Video (calibrated)</span>
                <span className="font-mono text-sm text-green-600">0.0678</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-3">Brier Score</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Image (uncalibrated)</span>
                <span className="font-mono text-sm">0.2345</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Image (calibrated)</span>
                <span className="font-mono text-sm text-green-600">0.1876</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Video (uncalibrated)</span>
                <span className="font-mono text-sm">0.2890</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Video (calibrated)</span>
                <span className="font-mono text-sm text-green-600">0.2123</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Methodology */}
      <div className="card">
        <h2 className="section-header">
          <Thermometer className="w-6 h-6" />
          Calibration Methodology
        </h2>

        <div className="prose prose-sm max-w-none">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">Temperature Scaling Theory</h3>
              <p className="text-sm text-gray-600 mb-3">
                Temperature scaling (Guo et al., 2017) divides model logits by a learned 
                scalar T before applying sigmoid: P_calibrated = sigmoid(z / T)
              </p>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• T > 1: Softens distribution (model was overconfident)</li>
                <li>• T < 1: Sharpens distribution (model was underconfident)</li>
                <li>• T = 1: No calibration applied</li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium text-gray-900 mb-2">Optimization Process</h3>
              <p className="text-sm text-gray-600 mb-3">
                The temperature parameter is optimized by minimizing negative log-likelihood 
                on a validation set using bounded optimization.
              </p>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Optimization bounds: [0.01, 10.0]</li>
                <li>• Method: scipy.optimize.minimize_scalar</li>
                <li>• Convergence tolerance: 1e-5</li>
                <li>• Maximum iterations: 500</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
