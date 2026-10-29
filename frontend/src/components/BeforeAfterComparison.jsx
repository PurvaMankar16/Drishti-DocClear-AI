import { useMemo } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Summarises a list of enhancement operations into human-readable labels.
 * Only operations that were actually applied are included.
 */
function AppliedOperations({ operations }) {
  if (!operations?.length) return null;
  const applied = operations.filter((op) => op.applied);
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
        </li>
      ))}
    </ul>
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
  const enhancedUrl = result?.enhanced_file?.download_url
    ? `${API_BASE_URL}${result.enhanced_file.download_url}`
    : null;

  const recommended = result?.final_recommendation === 'recommended';
  const reviewRequired = result?.final_recommendation === 'review_required';
  const ocrUnavailable = result?.final_recommendation === 'ocr_validation_unavailable';
  const readability = result?.ocr_validation;

  return (
    <section className="comparison-panel">
      {/* ── heading ── */}
      <div className="comparison-heading">
        <div>
          <p className="eyebrow">Result</p>
          <h2>
            {recommended
              ? 'Enhanced document ready'
              : reviewRequired
              ? 'Review recommended — enhancement produced mixed results'
              : 'Original document preserved'}
          </h2>
        </div>
        <p className={recommended ? 'success-copy' : 'notice-copy'}>
          {recommended
            ? 'The enhanced derivative improved measured quality and OCR readability was validated.'
            : reviewRequired
            ? 'The quality score improved, but OCR validation did not confirm readability preservation. Review both versions before proceeding.'
            : ocrUnavailable
            ? 'OCR validation could not run. The enhanced image was produced but cannot be automatically recommended.'
            : 'Automatic enhancement could not sufficiently improve this document.'}
        </p>
      </div>

      {/* ── side-by-side images ── */}
      <div className="comparison-grid">
        <figure>
          <figcaption>Original · {result.original_quality.overall_score} / 100</figcaption>
          {originalFile?.type === 'application/pdf' ? (
            <iframe title="Original document" src={originalUrl} />
          ) : (
            <img src={originalUrl} alt="Original document" />
          )}
        </figure>
        <figure>
          <figcaption>Enhanced · {result.enhanced_quality.overall_score} / 100</figcaption>
          {recommended ? (
            <img src={enhancedUrl} alt="Enhanced document" />
          ) : (
            <div className="unavailable-preview">
              {reviewRequired || ocrUnavailable
                ? 'Enhanced image produced but not recommended — download link is unavailable until validation passes.'
                : 'No enhanced result recommended.'}
            </div>
          )}
        </figure>
      </div>

      {/* ── improvement numbers ── */}
      <div className="improvement-row">
        <span>Measured improvement</span>
        <strong>
          {result.improvement.absolute_score_change > 0 ? '+' : ''}
          {result.improvement.absolute_score_change} points
        </strong>
      </div>

      {/* ── applied operations list ── */}
      <div className="operations-panel">
        <p className="eyebrow">Enhancement operations applied</p>
        <AppliedOperations operations={result.operations} />
      </div>

      {/* ── OCR readability panel ── */}
      <div className="readability-panel">
        <p className="eyebrow">OCR readability validation</p>

        {readability?.available ? (
          <>
            {/* Metrics grid — only when both snapshots exist */}
            {readability.original && readability.enhanced && (
              <div className="readability-grid">
                <div>
                  <span>Original confidence</span>
                  <strong>{Math.round(readability.original.confidence * 100)}%</strong>
                </div>
                <div>
                  <span>Enhanced confidence</span>
                  <strong>{Math.round(readability.enhanced.confidence * 100)}%</strong>
                </div>
                <div>
                  <span>Confidence delta</span>
                  <strong>
                    {readability.improvement?.confidence_delta > 0 ? '+' : ''}
                    {Math.round((readability.improvement?.confidence_delta || 0) * 100)} pts
                  </strong>
                </div>
                <div>
                  <span>Detected words</span>
                  <strong>
                    {readability.original.word_count} → {readability.enhanced.word_count}
                  </strong>
                </div>
                {readability.text_similarity !== null && readability.text_similarity !== undefined && (
                  <div>
                    <span>Text preservation</span>
                    <strong>{Math.round(readability.text_similarity * 100)}%</strong>
                  </div>
                )}
              </div>
            )}

            {/* Status message */}
            {readability.status === 'validated' && (
              <p className="success-copy">
                OCR validation passed: text readability improved and document content was preserved.
              </p>
            )}
            {readability.status === 'rejected' && (
              <p className="notice-copy">
                {readability.content_preservation_warning
                  ? 'OCR validation failed: enhancement altered detected text or reduced content similarity below the safe threshold.'
                  : 'OCR validation failed: enhancement degraded readability or reduced the amount of recognised text.'}
              </p>
            )}
            {readability.status === 'inconclusive' && (
              <p className="notice-copy">
                OCR validation was inconclusive: too little text was recognised in either version to
                measure a reliable readability change. This is normal for dense handwriting. Review
                the images visually before deciding.
              </p>
            )}
            {readability.error && (
              <p className="notice-copy">{readability.error}</p>
            )}

            {/* Detected text details */}
            <details>
              <summary>View detected text</summary>
              <div className="detected-text">
                <p>
                  <strong>Original:</strong>{' '}
                  {readability.original_text || 'No text detected.'}
                </p>
                <p>
                  <strong>Enhanced:</strong>{' '}
                  {readability.enhanced_text || 'No text detected.'}
                </p>
              </div>
            </details>
          </>
        ) : (
          <p className="notice-copy">
            {readability?.error
              ? `OCR validation could not run: ${readability.error}`
              : 'OCR validation is unavailable. Enhancement cannot be automatically recommended until the OCR runtime is set up.'}
          </p>
        )}
      </div>

      {/* ── download link (recommended only) ── */}
      {recommended && (
        <a
          className="download-link"
          href={enhancedUrl}
          download={result.enhanced_file.filename}
        >
          Download enhanced document
        </a>
      )}

      <p className="preservation-note">Original document remains unchanged.</p>
    </section>
  );
}
