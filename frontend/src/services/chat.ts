import { apiRequest } from './api'
export type SourceNode = { text?: string; content?: string; score?: number | null; source?: string | null; metadata?: Record<string, unknown> }
export type QueryResponse = { answer: string; query?: string | null; retrieved_nodes?: SourceNode[]; source_nodes?: SourceNode[]; routing?: string; validation_results?: { input_allowed: boolean; output_blocked?: boolean; output_sanitized?: boolean; output_block_reason?: string | null } | null }
export function submitQuery(token: string, question: string, userEmail: string) { return apiRequest<QueryResponse>('/query', token, { method: 'POST', body: JSON.stringify({ question, user_email: userEmail }) }) }
