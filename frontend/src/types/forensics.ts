export interface ForensicsResult {
  prediction: 'real' | 'fake';
  probability: number;
  confidence: number;
  modality: 'image' | 'video' | 'audio';
  inference_latency_ms: number;
  explanation_metadata?: Record<string, unknown>;
}
