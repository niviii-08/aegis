import { useState, useRef } from 'react'
import { apiClient, PredictionResponse } from '@/lib/api'
import { Upload, Layers, X, AlertCircle, CheckCircle, Clock, Image as ImageIcon, Video, Mic } from 'lucide-react'
import { formatProbability, formatLatency } from '@/lib/utils'

export default function MultimodalDetection() {
  const [files, setFiles] = useState<{ image?: File; video?: File; audio?: File }>({})
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (type: 'image' | 'video' | 'audio', selectedFile: File) => {
    setFiles(prev => ({ ...prev, [type]: selectedFile }))
    setError(null)
    setResult(null)
  }

  const handleDrop = (e: React.DragEvent, type: 'image' | 'video' | 'audio') => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    const mimeTypes = {
      image: 'image/',
      video: 'video/',
      audio: 'audio/'
    }
    
    if (droppedFile && droppedFile.type.startsWith(mimeTypes[type])) {
      handleFileSelect(type, droppedFile)
    } else {
      setError(`Please upload a valid ${type} file`)
    }
  }

  const handleFileChange = (type: 'image' | 'video' | 'audio', e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      handleFileSelect(type, selectedFile)
    }
  }

  const handlePredict = async () => {
    if (Object.keys(files).length === 0) {
      setError('Please select at least one file')
      return
    }

    try {
      setPredicting(true)
      setError(null)
      const prediction = await apiClient.predictMultimodal(files)
      setResult(prediction)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed')
      console.error(err)
    } finally {
      setPredicting(false)
    }
  }

  const handleReset = () => {
    setFiles({})
    setResult(null)
    setError(null)
    if (imageInputRef.current) imageInputRef.current.value = ''
    if (videoInputRef.current) videoInputRef.current.value = ''
    if (audioInputRef.current) audioInputRef.current.value = ''
  }

  const removeFile = (type: 'image' | 'video' | 'audio') => {
    setFiles(prev => {
      const newFiles = { ...prev }
      delete newFiles[type]
      return newFiles
    })
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Multimodal Detection</h1>
        <p className="text-gray-600">
          Combine multiple modalities for enhanced deepfake detection using fusion models
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="card">
          <h2 className="section-header">
            <Layers className="w-6 h-6" />
            Upload Files
          </h2>

          <div className="space-y-4">
            {/* Image Upload */}
            <div
              className={`upload-zone ${files.image ? 'border-primary-400 bg-primary-50' : ''}`}
              onDrop={(e) => handleDrop(e, 'image')}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => !files.image && imageInputRef.current?.click()}
            >
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => handleFileChange('image', e)}
                className="hidden"
              />
              {files.image ? (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <ImageIcon className="w-8 h-8 text-primary-600" />
                    <div>
                      <div className="font-medium text-gray-900">{files.image.name}</div>
                      <div className="text-sm text-gray-600">
                        {(files.image.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeFile('image'); }}
                    className="p-2 hover:bg-white rounded-full"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <ImageIcon className="w-8 h-8 mx-auto text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">Add image (optional)</p>
                </div>
              )}
            </div>

            {/* Video Upload */}
            <div
              className={`upload-zone ${files.video ? 'border-primary-400 bg-primary-50' : ''}`}
              onDrop={(e) => handleDrop(e, 'video')}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => !files.video && videoInputRef.current?.click()}
            >
              <input
                ref={videoInputRef}
                type="file"
                accept="video/*"
                onChange={(e) => handleFileChange('video', e)}
                className="hidden"
              />
              {files.video ? (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Video className="w-8 h-8 text-primary-600" />
                    <div>
                      <div className="font-medium text-gray-900">{files.video.name}</div>
                      <div className="text-sm text-gray-600">
                        {(files.video.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeFile('video'); }}
                    className="p-2 hover:bg-white rounded-full"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <Video className="w-8 h-8 mx-auto text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">Add video (optional)</p>
                </div>
              )}
            </div>

            {/* Audio Upload */}
            <div
              className={`upload-zone ${files.audio ? 'border-primary-400 bg-primary-50' : ''}`}
              onDrop={(e) => handleDrop(e, 'audio')}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => !files.audio && audioInputRef.current?.click()}
            >
              <input
                ref={audioInputRef}
                type="file"
                accept="audio/*"
                onChange={(e) => handleFileChange('audio', e)}
                className="hidden"
              />
              {files.audio ? (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Mic className="w-8 h-8 text-primary-600" />
                    <div>
                      <div className="font-medium text-gray-900">{files.audio.name}</div>
                      <div className="text-sm text-gray-600">
                        {(files.audio.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); removeFile('audio'); }}
                    className="p-2 hover:bg-white rounded-full"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <div className="text-center">
                  <Mic className="w-8 h-8 mx-auto text-gray-400 mb-2" />
                  <p className="text-sm text-gray-600">Add audio (optional)</p>
                </div>
              )}
            </div>

            <div className="flex gap-2">
              <button
                onClick={handlePredict}
                disabled={predicting || Object.keys(files).length === 0}
                className="btn-primary flex-1"
              >
                {predicting ? (
                  <span className="flex items-center justify-center gap-2">
                    <Clock className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </span>
                ) : (
                  'Run Multimodal Analysis'
                )}
              </button>
              <button
                onClick={handleReset}
                disabled={predicting}
                className="btn-secondary"
              >
                Reset
              </button>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="card">
          <h2 className="section-header">
            <CheckCircle className="w-6 h-6" />
            Fusion Results
          </h2>

          {!result ? (
            <div className="flex items-center justify-center h-64 text-gray-400">
              <div className="text-center">
                <Layers className="w-12 h-12 mx-auto mb-4" />
                <p>Upload at least one file to see fusion results</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Active Modalities */}
              <div className="flex flex-wrap gap-2">
                {Object.keys(files).map(modality => (
                  <span key={modality} className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium capitalize">
                    {modality}
                  </span>
                ))}
              </div>

              {/* Main Prediction */}
              <div className="text-center p-6 bg-gradient-to-br from-research-purple to-research-blue rounded-lg text-white">
                <div className="text-sm opacity-80 mb-2">Fused Prediction</div>
                <div className="text-4xl font-bold mb-2">
                  {result.prediction.toUpperCase()}
                </div>
                <div className="text-sm opacity-80">
                  Model: {result.model_version}
                </div>
              </div>

              {/* Confidence Metrics */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Fused Confidence</span>
                    <span className="font-medium">{formatProbability(result.probability)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-research-purple h-2 rounded-full transition-all duration-500"
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
                  <div className="text-sm text-gray-600 mb-1">Active Modalities</div>
                  <div className="text-xl font-bold">{Object.keys(files).length}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Research Context */}
      <div className="card">
        <h2 className="section-header">
          <Layers className="w-6 h-6" />
          Multimodal Fusion Research
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Fusion Architecture</h3>
            <p className="text-sm text-gray-600">
              The multimodal system uses learned gating networks to optimally combine
              predictions from individual modality classifiers based on input characteristics.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Cross-Modal Consistency</h3>
            <p className="text-sm text-gray-600">
              Analyzes consistency across modalities to detect discrepancies that may
              indicate manipulation, leveraging complementary information sources.
            </p>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Research Status</h3>
            <p className="text-sm text-gray-600">
              Current implementation provides single-modality fallback. Full fusion
              architecture is under development for enhanced detection accuracy.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
