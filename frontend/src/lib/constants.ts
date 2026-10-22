export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const SUPPORTED_EXTENSIONS = {
  image: ['.jpg', '.jpeg', '.png', '.webp'],
  audio: ['.wav', '.flac', '.mp3', '.npy'],
  video: ['.mp4', '.avi', '.mov', '.webm'],
};
