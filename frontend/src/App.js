import React, { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleUpload = async () => {
    if (!file) return alert('Select a video first');
    setLoading(true);
    setResult(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: form
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Server error');
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert('Upload error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Violence Detection Demo</h1>
      <input type="file" accept="video/*" onChange={handleFile} />
      { preview && <video src={preview} controls width="400" style={{display:'block', marginTop:10}} /> }
      <div style={{marginTop:10}}>
        <button onClick={handleUpload} disabled={loading}>{loading ? 'Analyzing...' : 'Analyze'}</button>
      </div>
      {result && (
        <div className="result">
          <h3>Result</h3>
          <p>Label: <strong>{result.label}</strong></p>
          <p>Violence score: <strong>{(result.violence_score).toFixed(3)}</strong></p>
        </div>
      )}
    </div>
  );
}
