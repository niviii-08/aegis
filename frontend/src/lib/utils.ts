export function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ')
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
      return 'text-green-600'
    case 'unhealthy':
    case 'load_failed':
      return 'text-red-600'
    case 'warning':
    case 'checkpoint_not_found':
      return 'text-yellow-600'
    default:
      return 'text-gray-600'
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
      return 'bg-gray-100 text-gray-800'
  }
}
