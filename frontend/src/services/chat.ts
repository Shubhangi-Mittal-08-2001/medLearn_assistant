import { apiRequest, ApiError } from './api'
export type SourceNode = { text?: string; content?: string; score?: number | null; source?: string | null; metadata?: Record<string, unknown> }
export type QueryResponse = { answer: string; query?: string | null; retrieved_nodes?: SourceNode[]; source_nodes?: SourceNode[]; routing?: string; validation_results?: { input_allowed: boolean; output_blocked?: boolean; output_sanitized?: boolean; output_block_reason?: string | null } | null }
export function submitQuery(token: string, question: string, userEmail: string) { return apiRequest<QueryResponse>('/query', token, { method: 'POST', body: JSON.stringify({ question, user_email: userEmail }) }) }
export async function submitQueryStream(token: string, question: string, userEmail: string, onChunk: (chunk: string) => void) {
	let response: Response
	try {
		response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/query/stream`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
			body: JSON.stringify({ question, user_email: userEmail }),
		})
	} catch { throw new ApiError('MedLearn is unavailable right now. Please try again shortly.', 0) }
	if (response.status === 401) throw new ApiError('Your session has expired. Please sign in again.', 401)
	if (!response.ok || !response.body) throw new ApiError('Something went wrong. Please try again.', response.status)
	const reader = response.body.getReader()
	const decoder = new TextDecoder()
	let buffer = ''
	let completed: QueryResponse | undefined
	while (true) {
		const { value, done } = await reader.read()
		buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
		const events = buffer.split('\n\n')
		buffer = events.pop() || ''
		for (const event of events) {
			const data = event.split('\n').find((line) => line.startsWith('data: '))?.slice(6)
			if (!data) continue
			const payload = JSON.parse(data) as { content?: string } & Partial<QueryResponse>
			if (event.startsWith('event: chunk')) onChunk(payload.content || '')
			if (event.startsWith('event: done')) completed = payload as QueryResponse
		}
		if (done) break
	}
	return completed || { answer: '', source_nodes: [] }
}
