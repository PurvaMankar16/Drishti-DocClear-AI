import { useEffect, useRef, useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Displays document layout detection results with bounding boxes.
 * Shows detected regions like text blocks, tables, figures, headers, etc.
 */
export default function LayoutDetectionDisplay({ originalFile, layoutResult, documentId }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Color mapping for different layout elements
  const layoutColors = {
    'Text': '#3b82f6',        // blue
    'Title': '#8b5cf6',       // purple
    'Section-header': '#6366f1', // indigo
    'Page-header': '#10b981',  // green
    'Page-footer': '#14b8a6',  // teal
    'Caption': '#f59e0b',      // amber
    'Footnote': '#ef4444',     // red
    'Table': '#ec4899',        // pink
    'Picture': '#f97316',      // orange
    'Formula': '#06b6d4',      // cyan
    'List-item': '#84cc16',    // lime
  };

  const getColor = (label) => layoutColors[label] || '#64748b'; // default gray

  useEffect(() => {
    if (!imageLoaded || !layoutResult || !canvasRef.current || !imageRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;

    // Set canvas size to match image display size
    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes for each page
    layoutResult.pages?.forEach(page => {
      const scaleX = canvas.width / page.image_width;
      const scaleY = canvas.height / page.image_height;

      page.detections?.forEach(detection => {
        const { bbox, label, confidence } = detection;
        const color = getColor(label);

        // Scale coordinates to canvas size
        const x1 = bbox.x1 * scaleX;
        const y1 = bbox.y1 * scaleY;
        const width = (bbox.x2 - bbox.x1) * scaleX;
        const height = (bbox.y2 - bbox.y1) * scaleY;

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        // Draw label background
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.9;
        const labelText = `${label} (${(confidence * 100).toFixed(0)}%)`;
        const textMetrics = ctx.measureText(labelText);
        const labelHeight = 18;
        const labelWidth = textMetrics.width + 8;
        ctx.fillRect(x1, y1 - labelHeight, labelWidth, labelHeight);

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.globalAlpha = 1.0;
        ctx.font = '12px sans-serif';
        ctx.fillText(labelText, x1 + 4, y1 - 5);
      });
    });
  }, [imageLoaded, layoutResult]);

  // Use backend preview endpoint that renders PDF pages as PNG
  const imageUrl = documentId
    ? `${API_BASE_URL}/api/documents/${documentId}/preview?page=1`
    : originalFile
      ? URL.createObjectURL(originalFile)
      : null;

  const hasDetections = layoutResult?.status === 'detected' && layoutResult?.total_detections > 0;

  return (
    <section className="layout-detection-panel">
      <div className="layout-heading">
        <div>
          <p className="eyebrow">Document Layout Detection</p>
          <h2>Detected Structure Elements</h2>
        </div>
        {layoutResult?.model_available ? (
          <p className="subtitle">
            Faster R-CNN with ResNet50-FPN · {layoutResult.total_detections} regions detected
          </p>
        ) : (
          <p className="notice-copy">
            Layout detection model is unavailable. {layoutResult?.error || ''}
          </p>
        )}
      </div>

      {hasDetections && (
        <>
          {/* Image with bounding boxes overlay */}
          <div className="layout-visualization">
            {imageUrl ? (
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="Document with layout detections"
                  style={{ maxWidth: '100%', display: 'block' }}
                  onLoad={() => setImageLoaded(true)}
                  onError={(e) => {
                    console.error('Failed to load document image');
                    console.error('Image URL:', imageUrl);
                    console.error('Error details:', e.target.error || 'Unknown error');
                  }}
                  crossOrigin="anonymous"
                />
                <canvas
                  ref={canvasRef}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    pointerEvents: 'none'
                  }}
                />
              </div>
            ) : (
              <p className="notice-copy">Document image not available</p>
            )}
          </div>

          {/* Detection summary */}
          <div className="detection-summary">
            <h3>Detected Elements</h3>
            <div className="detection-grid">
              {Object.entries(
                layoutResult.pages[0]?.detections?.reduce((acc, det) => {
                  acc[det.label] = (acc[det.label] || 0) + 1;
                  return acc;
                }, {}) || {}
              )
                .sort((a, b) => b[1] - a[1])
                .map(([label, count]) => (
                  <div key={label} className="detection-item">
                    <span style={{ color: getColor(label), fontWeight: 'bold' }}>■</span>
                    <span>{label}</span>
                    <strong>{count}</strong>
                  </div>
                ))}
            </div>
          </div>

          {/* Detailed detection list */}
          <details>
            <summary>View all detections</summary>
            <div className="detection-details">
              {layoutResult.pages?.map(page => (
                <div key={page.page_number}>
                  {page.detections?.map((detection, idx) => (
                    <div key={idx} className="detection-entry">
                      <span style={{ color: getColor(detection.label) }}>
                        {detection.label}
                      </span>
                      <span>{(detection.confidence * 100).toFixed(1)}%</span>
                      <span className="bbox-coords">
                        [{Math.round(detection.bbox.x1)}, {Math.round(detection.bbox.y1)}] →
                        [{Math.round(detection.bbox.x2)}, {Math.round(detection.bbox.y2)}]
                      </span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </details>

          {/* Model info */}
          <p className="model-info">
            Detection parameters: confidence ≥ {(layoutResult.confidence_threshold * 100).toFixed(0)}%,
            NMS IoU = {layoutResult.nms_threshold.toFixed(2)}
          </p>
        </>
      )}

      {layoutResult?.status === 'no_detections' && (
        <p className="notice-copy">
          No layout elements were detected with the current confidence threshold.
          This is normal for very low-quality images or non-document content.
        </p>
      )}
    </section>
  );
}
