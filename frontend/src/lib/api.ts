import axios from 'axios'

const API_BASE_URL = (import.meta as any).env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface HealthResponse {
  status: string
  version: string
  services: Record<string, string>
}

export interface ModelInfo {
  status: string
  checkpoint_path: string
  calibration_path: string | null
  model_type: string
  version: string
}

export interface ModelsResponse {
  models: Record<string, ModelInfo>
}

export interface PredictionResponse {
  prediction: 'fake' | 'real'
  probability: number
  calibrated_probability: number
  modality: string
  model_version: string
  inference_latency_ms: number
  calibrated: boolean
  temperature: number | null
  explanation_metadata: Record<string, any> | null
}

export interface ErrorResponse {
  error: string
  detail?: string
}

// API Functions
export const apiClient = {
  async getHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/health')
    return response.data
  },

  async getModels(): Promise<ModelsResponse> {
    const response = await api.get<ModelsResponse>('/models')
    return response.data
  },

  async predictImage(file: File): Promise<PredictionResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post<PredictionResponse>('/predict/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async predictVideo(file: File): Promise<PredictionResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post<PredictionResponse>('/predict/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async predictAudio(file: File): Promise<PredictionResponse> {
    const formData = new FormData()
    formData.append('file', file)
    const response = await api.post<PredictionResponse>('/predict/audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async predictMultimodal(files: {
    image?: File
    video?: File
    audio?: File
  }): Promise<PredictionResponse> {
    const formData = new FormData()
    if (files.image) formData.append('image', files.image)
    if (files.video) formData.append('video', files.video)
    if (files.audio) formData.append('audio', files.audio)
    
    const response = await api.post<PredictionResponse>('/predict/multimodal', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
}

export default apiClient
