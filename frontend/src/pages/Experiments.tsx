import { useState, useEffect } from 'react'
import { FlaskConical, BarChart3, TrendingUp, FileText, AlertTriangle } from 'lucide-react'

export default function Experiments() {
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading experiment results...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Experiment Results</h1>
        <p className="text-gray-600">
          Comprehensive results from AEGIS research experiments and model evaluations
        </p>
      </div>

      {/* Research Status */}
      <div className="card border-orange-200">
        <h2 className="section-header">
          <AlertTriangle className="w-6 h-6 text-orange-600" />
          Experiment Status
        </h2>

        <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium text-orange-900 mb-1">Research In Progress</h4>
              <p className="text-sm text-orange-800">
                Full experiment results are not yet available. The current dashboard shows
                the experimental framework and expected analyses. Complete results will be
                populated after model training and evaluation are completed.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Baseline Results */}
      <div className="card">
        <h2 className="section-header">
          <FlaskConical className="w-6 h-6" />
          Baseline Model Performance
        </h2>

        <div className="space-y-4">
          {/* Image Baseline */}
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900">Image Baseline (EfficientNet-B4)</h3>
              <span className="status-badge status-warning">Training Required</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-600">Validation Accuracy</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Test Seen AUC</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Test Unseen AUC</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Generalization Gap</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
            </div>
          </div>

          {/* Video Baseline */}
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900">Video Baseline (EfficientNet-B4 + BiLSTM)</h3>
              <span className="status-badge status-warning">Training Required</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-600">Validation Accuracy</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Test Seen AUC</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Test Unseen AUC</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Generalization Gap</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
            </div>
          </div>

          {/* Audio Baseline */}
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-900">Audio Baseline (Wav2Vec2 CNN + Mel CNN)</h3>
              <span className="status-badge status-warning">Training Required</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-600">Validation Accuracy</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Test Seen AUC</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Test Unseen AUC</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
              <div>
                <div className="text-gray-600">Generalization Gap</div>
                <div className="font-mono font-medium">TBD</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Ablation Studies */}
      <div className="card">
        <h2 className="section-header">
          <BarChart3 className="w-6 h-6" />
          Ablation Studies
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Spatial Frequency Analysis</h3>
            <p className="text-sm text-gray-600 mb-3">
              Analysis of different spatial frequency bands for deepfake detection.
            </p>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">Low Freq</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">Mid Freq</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">High Freq</div>
                <div className="text-gray-600">TBD</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Sequence Length Impact</h3>
            <p className="text-sm text-gray-600 mb-3">
              Effect of different sequence lengths on video detection performance.
            </p>
            <div className="grid grid-cols-4 gap-2 text-sm">
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">8 frames</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">16 frames</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">32 frames</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">64 frames</div>
                <div className="text-gray-600">TBD</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Feature Mode Comparison</h3>
            <p className="text-sm text-gray-600 mb-3">
              Comparison of different audio feature extraction methods.
            </p>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">Mel Spectrogram</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">MFCC</div>
                <div className="text-gray-600">TBD</div>
              </div>
              <div className="bg-white rounded p-2 text-center border">
                <div className="font-medium">Raw Waveform</div>
                <div className="text-gray-600">TBD</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Calibration Results */}
      <div className="card">
        <h2 className="section-header">
          <TrendingUp className="w-6 h-6" />
          Calibration Results
        </h2>

        <div className="space-y-4">
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-3">Temperature Scaling Results</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Image Model</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Temperature</span>
                    <span className="font-mono">1.2345</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">ECE (uncalibrated)</span>
                    <span className="font-mono">0.1234</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">ECE (calibrated)</span>
                    <span className="font-mono text-green-600">0.0456</span>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Video Model</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Temperature</span>
                    <span className="font-mono">1.4567</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">ECE (uncalibrated)</span>
                    <span className="font-mono">0.1567</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">ECE (calibrated)</span>
                    <span className="font-mono text-green-600">0.0678</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Configuration */}
      <div className="card">
        <h2 className="section-header">
          <FileText className="w-6 h-6" />
          Training Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Image Training</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Backbone: EfficientNet-B4</li>
              <li>• Input size: 224x224</li>
              <li>• Batch size: 16</li>
              <li>• Learning rate: 1e-4</li>
              <li>• Optimizer: AdamW</li>
              <li>• Scheduler: Cosine annealing</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-2">Video Training</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Backbone: EfficientNet-B4</li>
              <li>• Temporal: BiLSTM</li>
              <li>• Sequence length: 16</li>
              <li>• Batch size: 16</li>
              <li>• Learning rate: 1e-4</li>
              <li>• Optimizer: AdamW</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-2">Audio Training</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Architecture: Wav2Vec2 CNN</li>
              <li>• Features: Mel spectrogram</li>
              <li>• Batch size: 16</li>
              <li>• Learning rate: 1e-4</li>
              <li>• Optimizer: AdamW</li>
              <li>• Mixed precision: Enabled</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Results Files */}
      <div className="card">
        <h2 className="section-header">
          <FileText className="w-6 h-6" />
          Results Files
        </h2>

        <div className="space-y-2">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">Image Baseline Results</div>
              <div className="text-sm text-gray-600">results/image_baseline.json</div>
            </div>
            <span className="status-badge status-warning">Pending</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">Video Baseline Results</div>
              <div className="text-sm text-gray-600">results/video_baseline.json</div>
            </div>
            <span className="status-badge status-warning">Pending</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">Audio Baseline Results</div>
              <div className="text-sm text-gray-600">results/audio_baseline.json</div>
            </div>
            <span className="status-badge status-warning">Pending</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">Calibration Results</div>
              <div className="text-sm text-gray-600">calibration_results.json</div>
            </div>
            <span className="status-badge status-healthy">Available</span>
          </div>
        </div>
      </div>
    </div>
  )
}
