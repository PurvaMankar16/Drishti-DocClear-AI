import { useRef, useState } from 'react';
import { analyzeDocument, enhanceDocument } from '../services/documentApi';
import BeforeAfterComparison from '../components/BeforeAfterComparison';

const ACCEPTED_TYPES = '.jpg,.jpeg,.png,.pdf';
const MAX_FILE_SIZE = 10 * 1024 * 1024;

function formatFileSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function statusLabel(status) {
  return status ? status.replace('_', ' ') : 'unknown';
}

function MetricCard({ label, value, status }) {
  return (
    <article className="metric-card">
      <div className="metric-card-heading">
        <h3>{label}</h3>
        <span className={`status status-${status}`}>{statusLabel(status)}</span>
      </div>
      <p>{value}</p>
    </article>
  );
}

export default function DocumentUpload() {
  const inputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [enhancementResult, setEnhancementResult] = useState(null);
  const [enhancementStage, setEnhancementStage] = useState('');

  function chooseFile(file) {
    setError('');
    setReport(null);
    setEnhancementResult(null);
    if (!file) return;
    const extension = `.${file.name.split('.').pop().toLowerCase()}`;
    if (!['.jpg', '.jpeg', '.png', '.pdf'].includes(extension)) {
      setSelectedFile(null);
      setError('Unsupported file. Please choose a JPG, JPEG, PNG, or PDF.');
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setSelectedFile(null);
      setError('This file is larger than the 10 MB limit.');
      return;
    }
    if (file.size === 0) {
      setSelectedFile(null);
      setError('The selected file is empty.');
      return;
    }
    setSelectedFile(file);
  }

  async function handleEnhance() {
    if (!report) return;
    setError('');
    setEnhancementStage('Applying document enhancement…');
    try {
      const result = await enhanceDocument(report.document_id);
      setEnhancementStage('Verifying improvement…');
      setEnhancementResult(result);
    } catch (enhancementError) {
      setError(enhancementError.message);
    } finally {
      setEnhancementStage('');
    }
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setError('Choose a document before analyzing it.');
      return;
    }
    setError('');
    setIsAnalyzing(true);
    try {
      setReport(await analyzeDocument(selectedFile));
    } catch (analysisError) {
      setError(analysisError.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  const metrics = report?.metrics;

  return (
    <main className="page-shell">
      <header className="page-header">
        <p className="eyebrow">DocClear AI</p>
        <h1>Intelligent Document Quality Assessment &amp; Enhancement</h1>
        <p className="subtitle">Upload a document to receive a clear quality baseline.</p>
      </header>

      <section
        className="upload-panel"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          chooseFile(event.dataTransfer.files[0]);
        }}
      >
        <input ref={inputRef} type="file" accept={ACCEPTED_TYPES} hidden onChange={(event) => chooseFile(event.target.files[0])} />
        <button className="drop-zone" type="button" onClick={() => inputRef.current?.click()}>
          <strong>Drop your document here</strong>
          <span>or click to browse · JPG, JPEG, PNG, PDF</span>
        </button>
        {selectedFile && (
          <div className="file-summary">
            <span>{selectedFile.name}</span>
            <span>{formatFileSize(selectedFile.size)}</span>
          </div>
        )}
        <button className="primary-button" type="button" disabled={!selectedFile || isAnalyzing} onClick={handleAnalyze}>
          {isAnalyzing ? 'Analyzing…' : 'Analyze Document'}
        </button>
        {error && <p className="error-message" role="alert">{error}</p>}
      </section>

      {report && metrics && (
        <section className="results-panel" aria-live="polite">
          <div className="score-card">
            <div>
              <p className="eyebrow">Overall quality score</p>
              <h2>{report.overall_score} <span>/ 100</span></h2>
            </div>
            <span className={`status status-${report.classification}`}>{statusLabel(report.classification)}</span>
          </div>
          <div className="decision-panel">
            {report.classification === 'acceptable' ? (
              <>
                <strong>Document quality is acceptable.</strong>
                <button className="secondary-button" type="button">Continue</button>
              </>
            ) : (
              <>
                <strong>{report.classification === 'poor' ? 'Document quality is poor. Automatic enhancement will be attempted.' : 'Document quality can potentially be improved.'}</strong>
                <button className="secondary-button" type="button" onClick={handleEnhance} disabled={Boolean(enhancementStage)}>
                  {enhancementStage || 'Enhance Document'}
                </button>
              </>
            )}
          </div>
          <div className="metrics-grid">
            <MetricCard label="Blur" value={`${metrics.blur.score}/100 · raw ${metrics.blur.raw_score.toFixed(1)}`} status={metrics.blur.status} />
            <MetricCard label="Brightness" value={`${metrics.brightness.mean_brightness.toFixed(1)} · ${statusLabel(metrics.brightness.condition)}`} status={metrics.brightness.status} />
            <MetricCard label="Contrast" value={`${metrics.contrast.score}/100 · raw ${metrics.contrast.raw_score.toFixed(1)}`} status={metrics.contrast.status} />
            <MetricCard label="Noise" value={`${metrics.noise.score}/100 · raw ${metrics.noise.raw_score.toFixed(1)}`} status={metrics.noise.status} />
            <MetricCard label="Skew" value={`${metrics.skew.angle.toFixed(1)}°`} status={metrics.skew.status} />
            <MetricCard label="Cropping" value={metrics.cropping.document_detected ? (metrics.cropping.centered ? 'Document centered' : 'Document detected') : 'Document boundary unclear'} status={metrics.cropping.status} />
            <MetricCard label="Resolution" value={`${metrics.resolution.width} × ${metrics.resolution.height} · ${metrics.resolution.megapixels.toFixed(2)} MP`} status={metrics.resolution.status} />
          </div>
          {enhancementResult && <BeforeAfterComparison originalFile={selectedFile} result={enhancementResult} />}
        </section>
      )}
    </main>
  );
}
