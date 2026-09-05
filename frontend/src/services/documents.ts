import { apiRequest } from './api'
export type UploadResponse = { message: string; documents_indexed: number }
export function uploadDocument(token: string, file: File) { const form = new FormData(); form.append('file', file); return apiRequest<UploadResponse>('/upload', token, { method: 'POST', body: form }) }
