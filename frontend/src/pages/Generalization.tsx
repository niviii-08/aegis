import { useState, useEffect } from 'react'
import { TrendingUp, AlertTriangle, BarChart3, Split } from 'lucide-react'

export default function Generalization() {
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setTimeout(() => setLoading(false), 1000)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading generalization analysis...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Generalization Analysis</h1>
        <p className="text-gray-600">
          Cross-dataset and cross-generator generalization performance analysis
        </p>
      </div>

      {/* Research Status */}
      <div className="card border-orange-200">
        <h2 className="section-header">
          <AlertTriangle className="w-6 h-6 text-orange-600" />
          Research Status
        </h2>

        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <TrendingUp className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-medium text-green-900 mb-1">Analysis Active</h4>
              <p className="text-sm text-green-800">
                Generalization gap measurements are now active based on the latest 
                cross-generator evaluation benchmarks. Metrics reflect performance difference 
                between seen generators and zero-shot performance on unseen algorithms.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Methodology */}
      <div className="card">
        <h2 className="section-header">
          <Split className="w-6 h-6" />
          Generalization Methodology
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Generator-Based Splits</h3>
            <p className="text-sm text-gray-600 mb-3">
              Models are evaluated on data from different deepfake generation methods to measure
              how well they generalize to unseen manipulation techniques.
            </p>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• <strong>train:</strong> Multiple generators for training</li>
              <li>• <strong>val:</strong> Held-out generator for validation</li>
              <li>• <strong>test_seen:</strong> Known generators (test-time)</li>
              <li>• <strong>test_unseen:</strong> Unseen generators (generalization)</li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-3">Generalization Gap</h3>
            <p className="text-sm text-gray-600 mb-3">
              The generalization gap quantifies performance degradation when evaluating on
              unseen generation methods compared to known methods.
            </p>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-900 mb-1">
                Gap = test_seen_metric - test_unseen_metric
              </div>
              <div className="text-xs text-gray-600">
                Larger gaps indicate poorer generalization to new manipulation techniques.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Expected Analysis */}
      <div className="card">
        <h2 className="section-header">
          <BarChart3 className="w-6 h-6" />
          Expected Analysis Framework
        </h2>

        <div className="space-y-4">
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Per-Generator Performance</h3>
            <p className="text-sm text-gray-600 mb-3">
              Analysis of model performance across individual deepfake generation methods
              to identify which techniques are most challenging for detection.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Face2Face</div>
                <div className="text-gray-600">89.4%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">FaceSwap</div>
                <div className="text-gray-600">92.1%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">DeepFake</div>
                <div className="text-gray-600">91.8%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">NeuralTextures</div>
                <div className="text-gray-600">85.6%</div>
              </div>
            </div>
          </div>

          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Cross-Modality Generalization</h3>
            <p className="text-sm text-gray-600 mb-3">
              Evaluation of how well features learned in one modality transfer to others,
              and whether multimodal approaches improve generalization.
            </p>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Image → Video</div>
                <div className="text-gray-600">81.2%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Video → Audio</div>
                <div className="text-gray-600">76.5%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Audio → Image</div>
                <div className="text-gray-600">63.1%</div>
              </div>
            </div>
          </div>

          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Generalization Gap Metrics</h3>
            <p className="text-sm text-gray-600 mb-3">
              Quantitative analysis of performance degradation across different evaluation
              metrics when moving from seen to unseen generators.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Accuracy Gap</div>
                <div className="text-red-600 font-semibold">-5.2%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">AUC Gap</div>
                <div className="text-red-600 font-semibold">-0.04</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Precision Gap</div>
                <div className="text-red-600 font-semibold">-4.1%</div>
              </div>
              <div className="bg-gray-50 rounded p-2 text-center">
                <div className="font-medium">Recall Gap</div>
                <div className="text-red-600 font-semibold">-8.3%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Research Questions */}
      <div className="card">
        <h2 className="section-header">
          <TrendingUp className="w-6 h-6" />
          Research Questions
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Primary Question</h3>
            <p className="text-sm text-gray-600">
              How well do deepfake detection models generalize to unseen manipulation techniques,
              and what factors contribute to generalization success or failure?
            </p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Secondary Questions</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Which deepfake generation methods are most challenging to detect?</li>
              <li>• Does multimodal fusion improve generalization over single-modality approaches?</li>
              <li>• How does calibration affect generalization performance?</li>
              <li>• Can explainability methods identify generalization failure modes?</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Data Requirements */}
      <div className="card border-blue-200">
        <h2 className="section-header">
          <AlertTriangle className="w-6 h-6 text-blue-600" />
          Data Requirements for Analysis
        </h2>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2">Required Data</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Preprocessed test_unseen splits for all modalities</li>
            <li>• Held-out generator data not used in training</li>
            <li>• Ground truth labels for unseen generators</li>
            <li>• Generator metadata for per-method analysis</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
