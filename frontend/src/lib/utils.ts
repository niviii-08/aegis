import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`
}

export function formatLatency(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`
  }
  return `${(ms / 1000).toFixed(2)}s`
}

export function formatNumber(num: number, decimals: number = 2): string {
  return num.toFixed(decimals)
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'loaded':
      return 'text-emerald-400'
    case 'unhealthy':
    case 'load_failed':
      return 'text-rose-400'
    case 'warning':
    case 'checkpoint_not_found':
      return 'text-amber-400'
    default:
      return 'text-gray-400'
  }
}

export function getStatusBadge(status: string): string {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'loaded':
      return 'status-healthy'
    case 'unhealthy':
    case 'load_failed':
      return 'status-unhealthy'
    case 'warning':
    case 'checkpoint_not_found':
      return 'status-warning'
    default:
      return 'bg-gray-800/80 text-gray-300 border-gray-700/50 border'
  }
}
