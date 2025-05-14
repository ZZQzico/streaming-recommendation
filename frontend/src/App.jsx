import { useState, useEffect } from 'react'
import axios from 'axios'

export default function App() {
  const [userId, setUserId] = useState('')
  const [datetime, setDatetime] = useState('')
  const [mode, setMode] = useState('single') // 'single' or 'batch'
  const [message, setMessage] = useState('')
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [flaggedItems, setFlaggedItems] = useState({}) // Track which items are flagged

  const handleSubmit = async e => {
    e.preventDefault()
    if (!userId || !datetime) {
      setMessage('❗ Please enter User ID and Time')
      return
    }
    // Convert to seconds timestamp
    const ts = Math.floor(new Date(datetime).getTime() / 1000)
    const url = mode === 'single'
      ? '/push_interest/'
      : '/send_kafka/'

    setLoading(true)
    setRecommendations([])
    setFlaggedItems({}) // Reset flagged items
    
    try {
      const res = await axios.post(url, { user_id: userId, timestamp: ts })
      if (res.data.status === 'success' || res.status === 200) {
        // Single user API has history_items_count, batch API only has message
        const info = res.data.history_items_count != null
          ? `Sent ${res.data.history_items_count} records`
          : res.data.message
        setMessage(`✅ [${mode === 'single' ? 'Single User' : 'Batch'}] ${info}`)
        
        // Wait 2 seconds after request, then try to fetch recommendations
        setTimeout(() => {
          fetchRecommendations(userId)
        }, 2000)
      } else {
        setMessage(`❌ Failed: ${res.data.message}`)
        setLoading(false)
      }
    } catch (err) {
      console.error(err)
      setMessage(`⚠️ Request error: ${err.response?.data?.detail || err.message}`)
      setLoading(false)
    }
  }

  const fetchRecommendations = async (uid) => {
    try {
      // Get recommendations from Redis
      const res = await axios.get(`/recommendations/${uid}`)
      if (res.data && res.data.recommendations) {
        setRecommendations(res.data.recommendations)
        setMessage(prevMessage => `${prevMessage} - Successfully retrieved recommendations`)
      } else {
        setMessage(prevMessage => `${prevMessage} - No recommendations found`)
      }
    } catch (err) {
      console.error(err)
      setMessage(prevMessage => `${prevMessage} - Failed to get recommendations: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleFlagClick = async (itemId) => {
    try {
      // Toggle the flag state
      const newFlaggedItems = { ...flaggedItems }
      newFlaggedItems[itemId] = !newFlaggedItems[itemId]
      setFlaggedItems(newFlaggedItems)
      
      // Send feedback to API
      await axios.post('/feedback/', {
        user_id: userId,
        item_id: itemId,
        is_satisfied: true // Always true when they click the flag button
      })
      
      console.log(`Feedback sent for item ${itemId}`)
    } catch (err) {
      console.error('Error sending feedback:', err)
      setMessage(prevMessage => `${prevMessage} - Error sending feedback: ${err.message}`)
    }
  }

  return (
    <div className="container">
      <h2>Recommendation System Demo</h2>
      <form onSubmit={handleSubmit}>
        <div className="row">
          <label>Mode:</label>
          <label><input
            type="radio" name="mode" value="single"
            checked={mode === 'single'}
            onChange={() => setMode('single')}
          /> Single User</label>
          <label><input
            type="radio" name="mode" value="batch"
            checked={mode === 'batch'}
            onChange={() => setMode('batch')}
          /> Batch</label>
        </div>
        <div className="row">
          <label>User ID:</label>
          <input
            value={userId}
            onChange={e => setUserId(e.target.value)}
            placeholder="Enter user_id"
          />
        </div>
        <div className="row">
          <label>Time:</label>
          <input
            type="datetime-local"
            value={datetime}
            onChange={e => setDatetime(e.target.value)}
          />
        </div>
        <button type="submit" className="btn" disabled={loading}>
          {loading ? 'Processing...' : (mode === 'single' ? 'Call /push_interest/' : 'Call /send_kafka/')}
        </button>
      </form>
      
      {message && <div className="message">{message}</div>}
      
      {loading && <div className="loading">Getting recommendations...</div>}
      
      {recommendations.length > 0 && (
        <div className="recommendations">
          <h3>Recommendations for user {userId}:</h3>
          <ul>
            {recommendations.map((item, index) => (
              <li key={index}>
                <span>{item}</span>
                <button 
                  className={`flag-btn ${flaggedItems[item] ? 'clicked' : ''}`}
                  onClick={() => handleFlagClick(item)}
                >
                  {flaggedItems[item] ? '✓ Liked' : '👍 Like'}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
