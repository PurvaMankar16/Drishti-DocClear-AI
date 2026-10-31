import { useMemo } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Summarises a list of enhancement operations into human-readable labels.
 * Only operations that were actually applied are included.
 */
function AppliedOperations({ operations }) {
  if (!operations?.length) return null;

  // Filter out validation operation for display
  const applied = operations.filter((op) => op.applied && op.name !== 'validation' && op.name !== 'pdf_processing');

  if (!applied.length) return <p className="notice-copy">No enhancement operations were applied.</p>;

  return (
    <ul className="applied-ops-list">
      {applied.map((op) => (
        <li key={op.name}>
          <strong>{formatOpName(op.name)}</strong>
          {op.parameters && (
            <span className="op-params">
              {' '}
              ({Object.entries(op.parameters)
                .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join('×') : v}`)
                .join(', ')})
            </span>
          )}
          {op.validation && op.validation.sharpness_improved && (
            <span className="validation-badge">✓ Validated</span>
          )}
          {op.validation && op.validation.contrast_improved && (
            <span className="validation-badge">✓ Validated</span>
          )}
        </li>
      ))}
    </ul>
  );
}

/**
 * Show validation results if available
 */
function ValidationResults({ operations }) {
  const validationOp = operations?.find(op => op.name === 'validation');
  if (!validationOp || !validationOp.result) return null;

  const { is_valid, message, details } = validationOp.result;

  return (
    <div className={`validation-results ${is_valid ? 'valid' : 'rejected'}`}>
      <h4>{is_valid ? '✓ Enhancement Validated' : '✗ Enhancement Rejected'}</h4>
      <p>{message}</p>
      {details && (
        <div className="validation-details">
          {details.blur_change_pct !== undefined && (
            <span className={details.blur_change_pct >= 0 ? 'positive' : 'negative'}>
              Sharpness: {details.blur_change_pct > 0 ? '+' : ''}{details.blur_change_pct}%
            </span>
          )}
          {details.contrast_change_pct !== undefined && (
            <span className={details.contrast_change_pct >= 0 ? 'positive' : 'negative'}>
              Contrast: {details.contrast_change_pct > 0 ? '+' : ''}{details.contrast_change_pct}%
            </span>
          )}
          {details.noise_change_pct !== undefined && (
            <span className={details.noise_change_pct >= 0 ? 'positive' : 'negative'}>
              Noise reduction: {details.noise_change_pct > 0 ? '+' : ''}{details.noise_change_pct}%
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function formatOpName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function BeforeAfterComparison({ originalFile, result }) {
  const originalUrl = useMemo(
    () => (originalFile ? URL.createObjectURL(originalFile) : null),
    [originalFile],
  );

  // Use preview URL for inline display
  const previewUrl = result?.enhanced_file?.preview_url
    ? `${API_BASE_URL}${result.enhanced_file.preview_url}`
    : null;

  // Use download URL only for the download button
  const downloadUrl = result?.enhanced_file?.download_url
    ? `${API_BASE_URL}${result.enhanced_file.download_url}`
    : null;

  const recommended = result?.final_recommendation === 'recommended';
  const rejected = result?.status === 'enhancement_rejected';
  const reviewRequired = result?.final_recommendation === 'review_required';

  // Determine file format
  const isOriginalPdf = originalFile?.type === 'application/pdf';
  const enhancedFormat = result?.enhanced_quality?.file_type === 'application/pdf' ? 'PDF' : 'Image';

  return (
    <section className="comparison-panel">
      {/* ── heading ── */}
      <div className="comparison-heading">
        <div>
          <p className="eyebrow">Result</p>
          <h2>
            {recommended
              ? `Enhanced ${enhancedFormat.toLowerCase()} ready`
              : rejected
              ? 'Enhancement rejected — original preserved'
              : reviewRequired
              ? 'Enhancement completed — review recommended'
              : 'Original document preserved'}
          </h2>
        </div>
        <p className={recommended ? 'success-copy' : rejected ? 'error-copy' : 'notice-copy'}>
          {recommended
            ? `The enhanced ${enhancedFormat.toLowerCase()} shows measurable quality improvement and is ready for use.`
            : rejected
            ? 'Enhancement would have degraded critical quality metrics (sharpness or contrast). Original document is better.'
            : reviewRequired
            ? 'Enhancement completed. Review both versions and compare the quality metrics before proceeding.'
            : 'Automatic enhancement could not sufficiently improve this document. Original quality may be acceptable as-is.'}
        </p>
      </div>

      {/* Validation results */}
      <ValidationResults operations={result.operations} />

      {/* ── side-by-side images ── */}
      <div className="comparison-grid">
        <figure>
          <figcaption>Original · {result.original_quality.overall_score} / 100</figcaption>
          {isOriginalPdf ? (
            <iframe title="Original document" src={originalUrl} />
          ) : (
            <img src={originalUrl} alt="Original document" />
          )}
        </figure>
        <figure>
          <figcaption>
            {recommended ? 'Enhanced' : rejected ? 'Rejected' : 'Enhanced'} · {result.enhanced_quality.overall_score} / 100
          </figcaption>
          {recommended && previewUrl ? (
            enhancedFormat === 'PDF' ? (
              <iframe title="Enhanced document preview" src={previewUrl} />
            ) : (
              <img src={previewUrl} alt="Enhanced document preview" />
            )
          ) : (
            <div className="unavailable-preview">
              {rejected
                ? 'Enhancement rejected. Original is better.'
                : reviewRequired
                ? 'Enhanced result not recommended for download.'
                : 'No enhanced result available.'}
            </div>
          )}
        </figure>
      </div>

      {/* ── improvement numbers with change indicators ── */}
      <div className="improvement-row">
        <span>Measured improvement</span>
        <strong className={result.improvement.absolute_score_change >= 0 ? 'positive' : 'negative'}>
          {result.improvement.absolute_score_change > 0 ? '+' : ''}
          {result.improvement.absolute_score_change} points
        </strong>
      </div>

      {/* ── quality metrics comparison with indicators ── */}
      <div className="metrics-panel">
        <p className="eyebrow">Quality metrics</p>
        <div className="metrics-grid">
          <div className={result.improvement.sharpness_change < 0 ? 'metric-degraded' : ''}>
            <span>Sharpness {result.improvement.sharpness_change < -10 ? '⚠️' : result.improvement.sharpness_change > 5 ? '✓' : ''}</span>
            <strong>
              {result.before.sharpness.toFixed(1)} → {result.after.sharpness.toFixed(1)}
              <span className={`change ${result.improvement.sharpness_change >= 0 ? 'positive' : 'negative'}`}>
                ({result.improvement.sharpness_change > 0 ? '+' : ''}{result.improvement.sharpness_change.toFixed(1)})
              </span>
            </strong>
          </div>
          <div>
            <span>Brightness</span>
            <strong>
              {Math.round(result.before.brightness)} → {Math.round(result.after.brightness)}
              <span className={`change ${result.improvement.brightness_change <= 0 ? 'positive' : 'neutral'}`}>
                ({result.improvement.brightness_change > 0 ? '+' : ''}{Math.round(result.improvement.brightness_change)})
              </span>
            </strong>
          </div>
          <div className={result.improvement.contrast_change < 0 ? 'metric-degraded' : ''}>
            <span>Contrast {result.improvement.contrast_change < -10 ? '⚠️' : result.improvement.contrast_change > 5 ? '✓' : ''}</span>
            <strong>
              {result.before.contrast.toFixed(1)} → {result.after.contrast.toFixed(1)}
              <span className={`change ${result.improvement.contrast_change >= 0 ? 'positive' : 'negative'}`}>
                ({result.improvement.contrast_change > 0 ? '+' : ''}{result.improvement.contrast_change.toFixed(1)})
              </span>
            </strong>
          </div>
          <div>
            <span>Noise level {result.improvement.noise_change > 10 ? '✓' : ''}</span>
            <strong>
              {result.before.noise.toFixed(1)} → {result.after.noise.toFixed(1)}
              <span className={`change ${result.improvement.noise_change >= 0 ? 'positive' : 'neutral'}`}>
                ({result.improvement.noise_change > 0 ? '+' : ''}{result.improvement.noise_change.toFixed(1)})
              </span>
            </strong>
          </div>
          <div>
            <span>Skew angle</span>
            <strong>
              {result.before.skew.toFixed(1)}° → {result.after.skew.toFixed(1)}°
            </strong>
          </div>
        </div>
      </div>

      {/* ── applied operations list ── */}
      <div className="operations-panel">
        <p className="eyebrow">Enhancement operations applied</p>
        <AppliedOperations operations={result.operations} />
      </div>

      {/* ── download link (recommended only) ── */}
      {recommended && downloadUrl && (
        <a
          className="download-link"
          href={downloadUrl}
          download={result.enhanced_file.filename}
        >
          Download Enhanced {enhancedFormat}
        </a>
      )}

      <p className="preservation-note">Original document remains unchanged.</p>
    </section>
  );
}
