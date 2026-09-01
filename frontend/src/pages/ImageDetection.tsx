import { useState, useRef } from 'react'
import { apiClient, PredictionResponse } from '@/lib/api'
import { Upload, Image as ImageIcon, X, AlertCircle, CheckCircle, Clock } from 'lucide-react'
import { formatProbability, formatLatency } from '@/lib/utils'

export default function ImageDetection() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile)
    setError(null)
    setResult(null)
    
    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result as string)
    }
    reader.readAsDataURL(selectedFile)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      handleFileSelect(droppedFile)
    } else {
      setError('Please upload an image file')
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      handleFileSelect(selectedFile)
    }
  }

  const handlePredict = async () => {
    if (!file) return

    try {
      setPredicting(true)
      setError(null)
      const prediction = await apiClient.predictImage(file)
      setResult(prediction)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed')
      console.error(err)
    } finally {
      setPredicting(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Image Detection</h1>
        <p className="text-gray-600">
          Detect deepfakes in images using spatial analysis and calibrated predictions
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="card">
          <h2 className="section-header">
            <ImageIcon className="w-6 h-6" />
            Upload Image
          </h2>

          {!file ? (
            <div
              className="upload-zone"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-600 mb-2">
                Drag and drop an image here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports: JPG, PNG, WebP, BMP (max 10MB)
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={preview || ''}
                  alt="Preview"
                  className="w-full h-64 object-contain bg-gray-100 rounded-lg"
                />
                <button
                  onClick={handleReset}
                  className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-md hover:bg-gray-100"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">{file.name}</span>
                <span className="text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>

              <button
                onClick={handlePredict}
                disabled={predicting}
                className="btn-primary w-full"
              >
                {predicting ? (
                  <span className="flex items-center justify-center gap-2">
                    <Clock className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </span>
                ) : (
                  'Detect Deepfake'
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="card">
          <h2 className="section-header">
            <CheckCircle className="w-6 h-6" />
            Analysis Results
          </h2>

          {!result ? (
            <div className="flex items-center justify-center h-64 text-gray-400">
              <div className="text-center">
                <ImageIcon className="w-12 h-12 mx-auto mb-4" />
                <p>Upload an image to see analysis results</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Main Prediction */}
              <div className="text-center p-6 bg-gradient-to-br from-primary-50 to-primary-100 rounded-lg">
                <div className="text-sm text-gray-600 mb-2">Prediction</div>
                <div className={`text-4xl font-bold mb-2 ${
                  result.prediction === 'fake' ? 'text-red-600' : 'text-green-600'
                }`}>
                  {result.prediction.toUpperCase()}
                </div>
                <div className="text-sm text-gray-600">
                  Model: {result.model_version}
                </div>
              </div>

              {/* Confidence Metrics */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Confidence</span>
                    <span className="font-medium">{formatProbability(result.probability)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${result.probability * 100}%` }}
                    />
                  </div>
                </div>

                {result.calibrated && (
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Calibrated Confidence</span>
                      <span className="font-medium">{formatProbability(result.calibrated_probability)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-research-green h-2 rounded-full transition-all duration-500"
                        style={{ width: `${result.calibrated_probability * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Temperature: {result.temperature?.toFixed(4)}
                    </div>
                  </div>
                )}
              </div>

              {/* Additional Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-1">Inference Time</div>
                  <div className="text-xl font-bold">{formatLatency(result.inference_latency_ms)}</div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-1">Modality</div>
                  <div className="text-xl font-bold capitalize">{result.modality}</div>
                </div>
              </div>

              {/* Explanation Metadata */}
              {result.explanation_metadata && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-900 mb-2">Analysis Details</div>
                  <div className="space-y-1 text-sm text-gray-600">
                    {Object.entries(result.explanation_metadata).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="font-mono">
                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Research Context */}
      <div className="card">
        <h2 className="section-header">
          <ImageIcon className="w-6 h-6" />
          Research Context
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Spatial Analysis</h3>
            <p className="text-sm text-gray-600">
              The image model uses EfficientNet-B4 backbone with spatial frequency analysis
              to detect manipulation artifacts in facial regions.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Calibration</h3>
            <p className="text-sm text-gray-600">
              Temperature scaling is applied to validation logits to produce well-calibrated
              probabilities that reflect true uncertainty.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Generalization</h3>
            <p className="text-sm text-gray-600">
              Model performance is evaluated across different deepfake generation methods
              to measure generalization to unseen manipulation techniques.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
