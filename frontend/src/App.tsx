import { useCallback, useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import './App.css'
import './auth.css'
import { useAuth0 } from '@auth0/auth0-react'
import { ApiError, setUnauthorizedHandler } from './services/api'
import { submitQueryStream, type QueryResponse, type SourceNode } from './services/chat'
import { uploadDocument } from './services/documents'

type Path = '/login' | '/chat' | '/admin'
type Session = { accessToken: string; user: { email: string; name: string; role: 'user' | 'admin' } }
type Message = { id: string; role: 'user' | 'assistant'; text: string; response?: QueryResponse }
type UploadState = { file: File; status: 'uploading' | 'success' | 'error'; message: string }
type IconName = 'logo'|'eye'|'eyeOff'|'send'|'upload'|'file'|'logout'|'plus'|'menu'|'close'|'shield'|'check'

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    logo: <><path d="M12 3v18M3 12h18"/><path d="M7 7h10v10H7z"/></>, eye: <><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.5"/></>, eyeOff: <><path d="m3 3 18 18M10.6 6.2A11 11 0 0 1 12 6c6.5 0 10 6 10 6a17 17 0 0 1-2 2.7M6.5 6.5C3.5 8.4 2 12 2 12s3.5 6 10 6a10 10 0 0 0 4-.8"/></>, send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>, upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/></>, file: <><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5"/></>, logout: <><path d="M10 17l5-5-5-5M15 12H3"/><path d="M14 3h7v18h-7"/></>, plus: <path d="M12 5v14M5 12h14"/>, menu: <path d="M4 7h16M4 12h16M4 17h16"/>, close: <path d="m6 6 12 12M18 6 6 18"/>, shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-5"/></>, check: <path d="m5 12 4 4L19 6"/>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}
function Brand({ admin = false }: { admin?: boolean }) { return <div className="brand"><span className="brand-mark"><Icon name="logo" size={18}/></span><span>MedLearn{admin && <em>Admin</em>}</span></div> }
function usePath(): [Path, (path: Path) => void] {
  const normalize = (): Path => ['/login','/chat','/admin'].includes(location.pathname) ? location.pathname as Path : '/login'
  const [path, setPath] = useState<Path>(normalize)
  useEffect(() => { const handler = () => setPath(normalize()); addEventListener('popstate', handler); return () => removeEventListener('popstate', handler) }, [])
  return [path, (next) => { history.replaceState({}, '', next); setPath(next) }]
}

function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [email,setEmail]=useState(''); const [password,setPassword]=useState(''); const [show,setShow]=useState(false); const [error]=useState(''); const [loading,setLoading]=useState(false)
  const submit=(event:FormEvent)=>{event.preventDefault();setLoading(true);onLogin()}
  return <main className="login-page"><section className="login-story"><Brand/><div className="story-copy"><span className="eyebrow">AI-powered medical learning</span><h1>Learn medicine with clarity and confidence.</h1><p>Explore complex concepts, search trusted course material, and turn your documents into a personal learning resource.</p><div className="trust-note"><span><Icon name="shield" size={20}/></span><div><strong>Grounded in your knowledge base</strong><small>Designed to support learning—not replace clinical judgment.</small></div></div></div><p className="story-footer">A focused space for curious medical minds.</p></section>
    <section className="login-panel"><div className="mobile-brand"><Brand/></div><form className="login-card" onSubmit={submit} noValidate><div><span className="eyebrow">Welcome back</span><h2>Sign in to MedLearn</h2><p>Continue to your medical learning workspace.</p></div><label>Email address<input autoComplete="username" type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} disabled={loading}/></label><label>Password<span className="password-wrap"><input autoComplete="current-password" type={show?'text':'password'} placeholder="Enter your password" value={password} onChange={e=>setPassword(e.target.value)} disabled={loading}/><button type="button" className="icon-button" onClick={()=>setShow(!show)} aria-label={show?'Hide password':'Show password'}><Icon name={show?'eyeOff':'eye'}/></button></span></label>{error&&<div className="form-error" role="alert">{error}</div>}<button className="primary-button" type="submit" disabled={loading}>{loading?<><span className="spinner"/>Signing in…</>:'Sign in'}</button><p className="privacy-note">Credentials are securely handled by your configured identity provider.</p></form></section></main>
}

function Markdown({text}:{text:string}) {
  const inline=(line:string)=>line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part,i)=>part.startsWith('**')?<strong key={i}>{part.slice(2,-2)}</strong>:part.startsWith('`')?<code key={i}>{part.slice(1,-1)}</code>:part)
  const nodes:ReactNode[]=[];let list:string[]=[];let ordered=false;const flush=()=>{if(list.length){const Tag=ordered?'ol':'ul';nodes.push(<Tag key={`l${nodes.length}`}>{list.map((item,i)=><li key={i}>{inline(item)}</li>)}</Tag>);list=[]}}
  text.split('\n').forEach(raw=>{const line=raw.trim();const match=line.match(/^([-*]|\d+\.)\s+(.+)/);if(match){const next=/\d/.test(match[1]);if(list.length&&ordered!==next)flush();ordered=next;list.push(match[2]);return}flush();if(!line)return;if(/^#{1,3}\s/.test(line))nodes.push(<h3 key={nodes.length}>{inline(line.replace(/^#{1,3}\s/,''))}</h3>);else nodes.push(<p key={nodes.length}>{inline(line)}</p>)});flush();return <div className="markdown">{nodes}</div>
}
function Sources({sources}:{sources:SourceNode[]}) { const unique=sources.filter((s,i,a)=>a.findIndex(x=>(x.source||x.metadata?.source||x.text)===(s.source||s.metadata?.source||s.text))===i);if(!unique.length)return null;return <div className="sources"><span className="sources-title">Sources</span>{unique.map((source,i)=><div className="source" key={i}><span><Icon name="file" size={16}/></span><div><strong>{String(source.source||source.metadata?.source||`Medical source ${i+1}`)}</strong></div></div>)}</div> }
function MessageView({message}:{message:Message}) { if(message.role==='user')return <div className="message user-message"><div>{message.text}</div></div>;const response=message.response;const ref=response?.answer.match(/Reference ID:\s*([A-Z0-9-]+)/i)?.[1];const handoff=response?.routing==='handoff'||Boolean(ref);const sources=response?.retrieved_nodes?.length?response.retrieved_nodes:response?.source_nodes||[];return <div className="message assistant-message"><div className="assistant-avatar"><Icon name="logo" size={15}/></div><div className="assistant-content"><span className="assistant-label">MedLearn</span>{!message.text&&!response?<><p>Analyzing your question</p><span className="typing"><i/><i/><i/></span></>:handoff?<div className="handoff"><span><Icon name="shield"/></span><div><strong>Expert review required</strong><p>This question needs further review to provide a reliable answer.</p>{ref&&<small>Reference ID <b>{ref}</b></small>}</div></div>:<Markdown text={message.text}/>}<Sources sources={sources}/>{response?.validation_results?.output_sanitized&&<div className="status-note">Sensitive information was removed for your privacy.</div>}</div></div> }

const suggestions=['Explain the somatosensory system in simple terms','Compare nociceptive and neuropathic pain','How does a reflex arc work?','Summarize a concept from my uploaded document']
function UploadCard({state,onDismiss,onRetry}:{state:UploadState;onDismiss:()=>void;onRetry:()=>void}) {return <div className={`upload-card ${state.status}`}><span className="file-icon"><Icon name={state.status==='success'?'check':'file'}/></span><div className="upload-copy"><strong>{state.file.name}</strong><small>{state.message}</small>{state.status==='uploading'&&<div className="progress"><span/></div>}</div>{state.status==='error'?<button onClick={onRetry}>Retry</button>:state.status!=='uploading'&&<button className="icon-button" onClick={onDismiss} aria-label="Dismiss upload"><Icon name="close" size={17}/></button>}</div>}

function ChatPage({session,onLogout}:{session:Session;onLogout:()=>void}) {
  const [messages,setMessages]=useState<Message[]>([]);const [question,setQuestion]=useState('');const [loading,setLoading]=useState(false);const [error,setError]=useState('');const [upload,setUpload]=useState<UploadState|null>(null);const [sidebar,setSidebar]=useState(false);const fileRef=useRef<HTMLInputElement>(null);const bottomRef=useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])
  const ask=async(value=question)=>{const clean=value.trim();if(!clean||loading)return;setMessages(old=>[...old,{id:crypto.randomUUID(),role:'user',text:clean}]);setQuestion('');setError('');setLoading(true);const assistantId=crypto.randomUUID();setMessages(old=>[...old,{id:assistantId,role:'assistant',text:''}]);try{const response=await submitQueryStream(session.accessToken,clean,session.user.email,(chunk)=>setMessages(old=>old.map(message=>message.id===assistantId?{...message,text:message.text+chunk}:message)));setMessages(old=>old.map(message=>message.id===assistantId?{...message,text:response.answer||message.text||'No answer was returned.',response}:message))}catch(err){setMessages(old=>old.filter(message=>message.id!==assistantId));if(!(err instanceof ApiError&&err.status===401))setError(err instanceof Error?err.message:'Your question could not be processed.')}finally{setLoading(false)}}
  const sendFile=async(file:File)=>{if(file.type!=='application/pdf'&&!file.name.toLowerCase().endsWith('.pdf'))return setUpload({file,status:'error',message:'Only PDF documents are supported.'});if(file.size>10*1024*1024)return setUpload({file,status:'error',message:'PDF must be smaller than 10 MB.'});setUpload({file,status:'uploading',message:'Processing document…'});try{const result=await uploadDocument(session.accessToken,file);setUpload({file,status:'success',message:`${result.message} ${result.documents_indexed} document${result.documents_indexed===1?'':'s'} indexed.`})}catch(err){setUpload({file,status:'error',message:err instanceof Error?err.message:'Upload failed.'})}}
  return <div className="app-shell"><header className="topbar"><button className="menu-button" onClick={()=>setSidebar(true)} aria-label="Open menu"><Icon name="menu"/></button><Brand/><div className="account"><div className="user-meta"><span>{session.user.name}</span><small>Student</small></div><span className="avatar">{session.user.name.charAt(0).toUpperCase()}</span><button className="logout-button" onClick={onLogout}><Icon name="logout" size={17}/><span>Log out</span></button></div></header><div className="workspace">{sidebar&&<button className="backdrop" aria-label="Close menu" onClick={()=>setSidebar(false)}/>}<aside className={sidebar?'sidebar open':'sidebar'}><div className="sidebar-mobile"><Brand/><button className="icon-button" onClick={()=>setSidebar(false)}><Icon name="close"/></button></div><button className="new-chat" onClick={()=>{setMessages([]);setError('');setSidebar(false)}}><Icon name="plus" size={18}/>New chat</button><nav><span className="nav-label">Workspace</span><button className="nav-item active"><span>✦</span>Chat</button><button className="nav-item" onClick={()=>fileRef.current?.click()}><Icon name="file" size={17}/>Upload document</button></nav><div className="sidebar-tip"><span><Icon name="shield" size={17}/></span><p><strong>Learning support</strong>Always verify medical information with trusted clinical guidance.</p></div></aside>
    <main className="chat-main"><div className="messages" aria-live="polite">{messages.length===0?<section className="empty-state"><div className="empty-logo"><Icon name="logo" size={25}/></div><span className="eyebrow">MedLearn Assistant</span><h1>What would you like to learn?</h1><p>Ask a medical question or explore concepts from your uploaded course material.</p><div className="suggestions">{suggestions.map(item=><button key={item} onClick={()=>ask(item)}><span>{item}</span><b>↗</b></button>)}</div></section>:<div className="message-list">{messages.map(message=><MessageView key={message.id} message={message}/>)}</div>}{loading&&<div className="message assistant-message loading-message"><div className="assistant-avatar"><Icon name="logo" size={15}/></div><div><span className="assistant-label">MedLearn</span><p>Analyzing your question</p><span className="typing"><i/><i/><i/></span></div></div>}{error&&<div className="query-error" role="alert"><span>!</span><p>{error}</p><button onClick={()=>setError('')}>Dismiss</button></div>}<div ref={bottomRef}/></div><div className="composer-area">{upload&&<UploadCard state={upload} onDismiss={()=>setUpload(null)} onRetry={()=>sendFile(upload.file)}/>}<form className="composer" onSubmit={e=>{e.preventDefault();ask()}}><button className="attach" type="button" onClick={()=>fileRef.current?.click()} disabled={loading} aria-label="Upload PDF"><Icon name="upload"/></button><textarea rows={1} value={question} maxLength={2000} placeholder="Ask a medical question…" onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}}} disabled={loading}/><button className="send-button" type="submit" disabled={!question.trim()||loading} aria-label="Send question"><Icon name="send" size={18}/></button></form><input ref={fileRef} hidden type="file" accept=".pdf,application/pdf" onChange={e=>{const file=e.target.files?.[0];if(file)sendFile(file);e.currentTarget.value=''}}/><small className="composer-hint">MedLearn can make mistakes. Verify information before applying it clinically.</small></div></main></div></div>
}
function AdminPage({session,onLogout}:{session:Session;onLogout:()=>void}) {return <div className="admin-page"><header className="admin-header"><Brand admin/><div className="account"><span className="avatar">{session.user.name.charAt(0).toUpperCase()}</span><button className="logout-button" onClick={onLogout}><Icon name="logout" size={17}/>Log out</button></div></header><main className="admin-content"><div className="admin-badge"><Icon name="shield" size={18}/>Administrator access</div><h1>Welcome, {session.user.name}</h1><p>You’re signed in with administrator access.</p><section className="admin-card"><div className="admin-card-icon"><Icon name="shield" size={26}/></div><div><h2>Administration workspace</h2><p>Your account is ready. Administrative tools can be added here when supported by the backend.</p></div></section></main></div>}

export default function App() {
  const { getAccessTokenSilently, isAuthenticated, isLoading, loginWithRedirect, logout: auth0Logout, user } = useAuth0()
  const [session, setSession] = useState<Session | null>(null)
  const [path, navigate] = usePath()
  const roleNamespace = import.meta.env.VITE_ROLE_NAMESPACE || 'https://stateful-agent.com/roles'
  const rawRoles = user?.[roleNamespace] ?? user?.[`${roleNamespace}/roles`] ?? user?.['https://stateful-agent.com/roles'] ?? []
  const roles = Array.isArray(rawRoles) ? rawRoles.map(String) : []
  const role: 'user' | 'admin' = roles.some((item) => item.toLowerCase() === 'admin') ? 'admin' : 'user'

  const logout = useCallback(() => {
    setSession(null)
    void auth0Logout({ logoutParams: { returnTo: window.location.origin } })
  }, [auth0Logout])

  useEffect(() => {
    setUnauthorizedHandler(logout)
    return () => {
      setUnauthorizedHandler(undefined)
    }
  }, [logout])

  useEffect(() => {
    if (!isAuthenticated || isLoading) return undefined
    let cancelled = false

    async function loadSession() {
      try {
        const accessToken = await getAccessTokenSilently()
        if (!cancelled) {
          setSession({
            accessToken,
            user: {
              email: user?.email || 'unknown',
              name: user?.name || user?.email || 'Authenticated user',
              role,
            },
          })
        }
      } catch {
        if (!cancelled) logout()
      }
    }

    void loadSession()
    return () => {
      cancelled = true
    }
  }, [getAccessTokenSilently, isAuthenticated, isLoading, logout, role, user])

  useEffect(() => {
    if (isLoading || !session) return
    const target: Path = session.user.role === 'admin' ? '/admin' : '/chat'
    if (path !== target) navigate(target)
  }, [isLoading, navigate, path, session])

  if (isLoading || (isAuthenticated && !session)) {
    return <main className="auth-callback"><span className="spinner dark"/><p>Completing secure sign-in…</p></main>
  }
  if (!isAuthenticated) return <LoginPage onLogin={() => void loginWithRedirect()}/>
  if (!session) return null
  return session.user.role === 'admin'
    ? <AdminPage session={session} onLogout={logout}/>
    : <ChatPage session={session} onLogout={logout}/>
}
