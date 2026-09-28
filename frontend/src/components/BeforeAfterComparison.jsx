import { useMemo } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export default function BeforeAfterComparison({ originalFile, result }) {
  const originalUrl = useMemo(() => (originalFile ? URL.createObjectURL(originalFile) : null), [originalFile]);
  const enhancedUrl = result?.enhanced_file?.download_url ? `${API_BASE_URL}${result.enhanced_file.download_url}` : null;
  const improved = result?.final_recommendation === 'recommended';
  const readability = result?.ocr_validation;

  return (
    <section className="comparison-panel">
      <div className="comparison-heading">
        <div>
          <p className="eyebrow">Result</p>
          <h2>{improved ? 'Enhanced document ready' : 'Original document preserved'}</h2>
        </div>
        <p className={improved ? 'success-copy' : 'notice-copy'}>
          {improved ? 'The enhanced derivative improved the measured quality.' : 'Automatic enhancement could not sufficiently improve this document.'}
        </p>
      </div>
      <div className="comparison-grid">
        <figure>
          <figcaption>Original · {result.original_quality.overall_score} / 100</figcaption>
          {originalFile?.type === 'application/pdf' ? <iframe title="Original document" src={originalUrl} /> : <img src={originalUrl} alt="Original document" />}
        </figure>
        <figure>
          <figcaption>Enhanced · {result.enhanced_quality.overall_score} / 100</figcaption>
          {improved ? <img src={enhancedUrl} alt="Enhanced document" /> : <div className="unavailable-preview">No enhanced result recommended</div>}
        </figure>
      </div>
      <div className="improvement-row">
        <span>Measured improvement</span>
        <strong>{result.improvement.absolute_score_change > 0 ? '+' : ''}{result.improvement.absolute_score_change} points</strong>
      </div>
      <div className="readability-panel">
        <p className="eyebrow">OCR readability validation</p>
        {readability?.available ? (
          <>
            <div className="readability-grid">
              <div><span>Original confidence</span><strong>{Math.round(readability.original.confidence * 100)}%</strong></div>
              <div><span>Enhanced confidence</span><strong>{Math.round(readability.enhanced.confidence * 100)}%</strong></div>
              <div><span>Confidence improvement</span><strong>+{Math.round(readability.improvement.confidence_delta * 100)} points</strong></div>
              <div><span>Detected words</span><strong>{readability.original.word_count} → {readability.enhanced.word_count}</strong></div>
              <div><span>Text preservation</span><strong>{Math.round(readability.text_similarity * 100)}%</strong></div>
            </div>
            <p className={readability.improvement.improved ? 'success-copy' : 'notice-copy'}>
              {readability.improvement.improved ? 'Readability improved.' : 'OCR readability did not improve sufficiently.'}
            </p>
            <details>
              <summary>View detected text</summary>
              <div className="detected-text"><p>{readability.original_text || 'No text detected.'}</p><p>{readability.enhanced_text || 'No text detected.'}</p></div>
            </details>
          </>
        ) : (
          <p className="notice-copy">OCR validation unavailable. Enhancement cannot be automatically recommended.</p>
        )}
      </div>
      {improved && <a className="download-link" href={enhancedUrl} download={result.enhanced_file.filename}>Download enhanced document</a>}
      <p className="preservation-note">Original document remains unchanged.</p>
    </section>
  );
}
