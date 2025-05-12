import { useState } from 'react'
import axios from 'axios'

export default function App() {
  const [userId, setUserId] = useState('')
  const [datetime, setDatetime] = useState('')
  const [mode, setMode] = useState('single') // 'single' 或 'batch'
  const [message, setMessage] = useState('')

  const handleSubmit = async e => {
    e.preventDefault()
    if (!userId || !datetime) {
      setMessage('❗ 请填写 用户ID 和 时间')
      return
    }
    // 转成秒级时间戳
    const ts = Math.floor(new Date(datetime).getTime() / 1000)
    const url = mode === 'single'
      ? '/push_interest/'
      : '/send_kafka/'

    try {
      const res = await axios.post(url, { user_id: userId, timestamp: ts })
      if (res.data.status === 'success' || res.status === 200) {
        // 单用户接口有 history_items_count，批量接口只有 message
        const info = res.data.history_items_count != null
          ? `已发送 ${res.data.history_items_count} 条记录`
          : res.data.message
        setMessage(`✅ [${mode === 'single' ? '单用户' : '批量'}] ${info}`)
      } else {
        setMessage(`❌ 失败：${res.data.message}`)
      }
    } catch (err) {
      console.error(err)
      setMessage(`⚠️ 请求出错：${err.response?.data?.detail || err.message}`)
    }
  }

  return (
    <div className="container">
      <h2>推送用户兴趣画像</h2>
      <form onSubmit={handleSubmit}>
        <div className="row">
          <label>模式：</label>
          <label><input
            type="radio" name="mode" value="single"
            checked={mode === 'single'}
            onChange={() => setMode('single')}
          /> 单用户</label>
          <label><input
            type="radio" name="mode" value="batch"
            checked={mode === 'batch'}
            onChange={() => setMode('batch')}
          /> 批量</label>
        </div>
        <div className="row">
          <label>用户 ID：</label>
          <input
            value={userId}
            onChange={e => setUserId(e.target.value)}
            placeholder="输入 user_id"
          />
        </div>
        <div className="row">
          <label>时间：</label>
          <input
            type="datetime-local"
            value={datetime}
            onChange={e => setDatetime(e.target.value)}
          />
        </div>
        <button type="submit" className="btn">
          {mode === 'single' ? '调用 /push_interest/' : '调用 /send_kafka/'}
        </button>
      </form>
      {message && <div className="message">{message}</div>}
    </div>
  )
}
