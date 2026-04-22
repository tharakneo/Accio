import { useState, useEffect } from 'react'
import './App.css'

const TMDB_IMG = 'https://image.tmdb.org/t/p/w200'
const TMDB_SEARCH = 'https://api.themoviedb.org/3/search/movie'
const TMDB_KEY = import.meta.env.VITE_TMDB_API_KEY

async function fetchPoster(title, year) {
  try {
    const res = await fetch(`${TMDB_SEARCH}?api_key=${TMDB_KEY}&query=${encodeURIComponent(title)}&year=${year}`)
    const data = await res.json()
    const poster = data.results?.[0]?.poster_path
    return poster ? `${TMDB_IMG}${poster}` : null
  } catch {
    return null
  }
}

function MovieCard({ movie, year, context, similarity, explanation }) {
  const [poster, setPoster] = useState(null)

  useEffect(() => {
    fetchPoster(movie, year).then(setPoster)
  }, [movie, year])

  return (
    <div className="card">
      {poster
        ? <img className="poster" src={poster} alt={movie} />
        : <div className="poster-placeholder">🎬</div>
      }
      <div className="card-info">
        <div className="card-title">{movie}</div>
        <div className="card-year">{year}</div>
        <div className="card-context">{explanation || `"${context}"`}</div>
        <div className="card-score">match · {Math.round((similarity || 0) * 100)}%</div>
      </div>
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)

  async function search() {
    if (!query.trim()) return
    setLoading(true)
    setSearched(true)
    try {
      const res = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      setResults(data.results || [])
    } catch {
      setResults([])
    }
    setLoading(false)
  }

  function handleKey(e) {
    if (e.key === 'Enter') search()
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Accio</h1>
        <p>Find any movie by describing it</p>
      </div>

      <div className="search-bar">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKey}
          placeholder="rom coms based in NYC..."
          autoFocus
        />
        <button onClick={search} disabled={loading}>
          {loading ? '...' : 'Search'}
        </button>
      </div>

      <div className="results">
        {loading && (
          <div className="loading">
            <div className="loading-dots">
              <span /><span /><span />
            </div>
          </div>
        )}
        {!loading && searched && results.length === 0 && (
          <div className="empty">No movies found. Try a different description.</div>
        )}
        {!loading && results.map((r, i) => (
          <MovieCard key={i} {...r} />
        ))}
      </div>
    </div>
  )
}
