export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) { super(message); this.status = status }
}

let onUnauthorized: (() => void) | undefined
export function setUnauthorizedHandler(handler?: () => void): void {
  onUnauthorized = handler
}

export async function apiRequest<T>(path: string, token: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers: { ...(init.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), Authorization: `Bearer ${token}`, ...init.headers } })
  } catch { throw new ApiError('MedLearn is unavailable right now. Please try again shortly.', 0) }
  if (response.status === 401) onUnauthorized?.()
  if (!response.ok) {
    const message = response.status === 401 ? 'Your session has expired. Please sign in again.' : response.status === 400 ? 'We could not process that request. Please review it and try again.' : 'Something went wrong. Please try again.'
    throw new ApiError(message, response.status)
  }
  return response.status === 204 ? undefined as T : response.json() as Promise<T>
}
