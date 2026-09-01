import { useState, useRef } from 'react'
import { apiClient, PredictionResponse } from '@/lib/api'
import { Upload, Mic, X, AlertCircle, CheckCircle, Clock } from 'lucide-react'
import { formatProbability, formatLatency } from '@/lib/utils'

export default function AudioDetection() {
  const [file, setFile] = useState<File | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile)
    setError(null)
    setResult(null)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      handleFileSelect(droppedFile)
    } else {
      setError('Please upload an audio file')
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
      const prediction = await apiClient.predictAudio(file)
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
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Audio Detection</h1>
        <p className="text-gray-600">
          Detect deepfakes in audio using spectral analysis and waveform modeling
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="card">
          <h2 className="section-header">
            <Mic className="w-6 h-6" />
            Upload Audio
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
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
              <p className="text-gray-600 mb-2">
                Drag and drop an audio file here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports: WAV, MP3, FLAC, OGG (max 50MB)
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="bg-gray-100 rounded-lg p-4 flex items-center gap-4">
                <Mic className="w-8 h-8 text-gray-400" />
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{file.name}</div>
                  <div className="text-sm text-gray-600">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                </div>
                <button
                  onClick={handleReset}
                  className="p-2 hover:bg-gray-200 rounded-full"
                >
                  <X className="w-4 h-4" />
                </button>
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
                <Mic className="w-12 h-12 mx-auto mb-4" />
                <p>Upload an audio file to see analysis results</p>
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
                  <div className="text-sm text-gray-600 mb-1">Feature Mode</div>
                  <div className="text-xl font-bold capitalize">
                    {result.explanation_metadata?.feature_mode || 'mel'}
                  </div>
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
          <Mic className="w-6 h-6" />
          Research Context
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Spectral Analysis</h3>
            <p className="text-sm text-gray-600">
              The audio model uses Wav2Vec2 CNN and Mel-spectrogram CNN architectures
              to detect spectral artifacts and waveform inconsistencies.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Feature Extraction</h3>
            <p className="text-sm text-gray-600">
              Supports multiple feature modes including Mel-spectrograms and raw waveform
              analysis for comprehensive audio deepfake detection.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Calibration</h3>
            <p className="text-sm text-gray-600">
              Audio-specific temperature scaling ensures calibrated probabilities across
              different audio quality levels and recording conditions.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
