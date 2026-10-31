import { useRef, useState } from 'react';
import { analyzeDocument, enhanceDocument, detectLayout } from '../services/documentApi';
import BeforeAfterComparison from '../components/BeforeAfterComparison';
import LayoutDetectionDisplay from '../components/LayoutDetectionDisplay';

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


function DefectPredictions({ predictions, modelAvailable }) {
  if (!modelAvailable || !predictions?.length) {
    return (
      <section className="defect-predictions">
        <h2>Deep-learning defect assessment</h2>
        <p>The defect classifier is unavailable. Classical quality analysis is still active.</p>
      </section>
    );
  }

  const labels = {
    blur: 'Blur',
    dark: 'Low brightness',
    low_contrast: 'Low contrast',
    noise: 'Noise',
    skew: 'Skew',
    low_resolution: 'Low resolution',
  };

  return (
    <section className="defect-predictions">
      <h2>Deep-learning defect assessment</h2>
      <p className="subtitle">Predictions from the ResNet-18 classifier. These are model estimates, not guarantees.</p>
      <div className="metrics-grid">
        {predictions.map((item) => (
          <article className="metric-card" key={item.label}>
            <div className="metric-card-heading">
              <h3>{labels[item.label] || item.label}</h3>
              <span className={`status status-${item.detected ? 'poor' : 'good'}`}>
                {item.detected ? 'Potential defect' : 'Not detected'}
              </span>
            </div>
            <p>{(item.probability * 100).toFixed(2)}% probability</p>
            <small>Decision threshold: {(item.threshold * 100).toFixed(0)}%</small>
          </article>
        ))}
      </div>
    </section>
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
  const [layoutResult, setLayoutResult] = useState(null);
  const [isDetectingLayout, setIsDetectingLayout] = useState(false);

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

  async function handleLayoutDetection() {
    if (!report) return;
    setError('');
    setIsDetectingLayout(true);
    try {
      const result = await detectLayout(report.document_id);
      setLayoutResult(result);
    } catch (layoutError) {
      setError(layoutError.message);
    } finally {
      setIsDetectingLayout(false);
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

          <div className="layout-detection-section">
            <h3>Document Structure Analysis</h3>
            <p>Detect layout elements like text blocks, tables, figures, headers, and more.</p>
            <button
              className="secondary-button"
              type="button"
              onClick={handleLayoutDetection}
              disabled={isDetectingLayout}
            >
              {isDetectingLayout ? 'Detecting Layout…' : 'Detect Document Layout'}
            </button>
          </div>

          <DefectPredictions predictions={report.defect_predictions} modelAvailable={report.ml_model_available} />
          <div className="metrics-grid">
            <MetricCard label="Blur" value={`${metrics.blur.score}/100 · raw ${metrics.blur.raw_score.toFixed(1)}`} status={metrics.blur.status} />
            <MetricCard label="Brightness" value={`${metrics.brightness.mean_brightness.toFixed(1)} · ${statusLabel(metrics.brightness.condition)}`} status={metrics.brightness.status} />
            <MetricCard label="Contrast" value={`${metrics.contrast.score}/100 · raw ${metrics.contrast.raw_score.toFixed(1)}`} status={metrics.contrast.status} />
            <MetricCard label="Noise" value={`${metrics.noise.score}/100 · raw ${metrics.noise.raw_score.toFixed(1)}`} status={metrics.noise.status} />
            <MetricCard label="Skew" value={`${metrics.skew.angle.toFixed(1)}°`} status={metrics.skew.status} />
            <MetricCard label="Cropping" value={metrics.cropping.document_detected ? (metrics.cropping.centered ? 'Document centered' : 'Document detected') : 'Document boundary unclear'} status={metrics.cropping.status} />
            <MetricCard label="Resolution" value={`${metrics.resolution.width} × ${metrics.resolution.height} · ${metrics.resolution.megapixels.toFixed(2)} MP`} status={metrics.resolution.status} />
          </div>
          {layoutResult && <LayoutDetectionDisplay originalFile={selectedFile} layoutResult={layoutResult} documentId={report.document_id} />}
          {enhancementResult && <BeforeAfterComparison originalFile={selectedFile} result={enhancementResult} />}
        </section>
      )}
    </main>
  );
}
