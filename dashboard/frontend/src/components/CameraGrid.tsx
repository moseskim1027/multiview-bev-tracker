import { useEffect, useRef } from "react";
import type { HomographyData, Track } from "../types";

interface Props {
  images: string[]; // base64 JPEG, indices 0-6 correspond to C1-C7
  selectedTrack: Track | undefined;
  homographyData: HomographyData | null;
  onShowCoverage: () => void;
}

const LABELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];

export default function CameraGrid({ images, selectedTrack, homographyData, onShowCoverage }: Props) {
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  useEffect(() => {
    canvasRefs.current.forEach((canvas, i) => {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!selectedTrack || canvas.width === 0) return;

      const bbox = selectedTrack.camera_bboxes?.[i];

      if (bbox && homographyData) {
        // Draw the actual detection bounding box, scaled from full-res to canvas size
        const scaleX = canvas.width / homographyData.orig_w;
        const scaleY = canvas.height / homographyData.orig_h;
        const x1 = bbox[0] * scaleX;
        const y1 = bbox[1] * scaleY;
        const w  = (bbox[2] - bbox[0]) * scaleX;
        const h  = (bbox[3] - bbox[1]) * scaleY;

        // Single colored bounding box
        ctx.strokeStyle = selectedTrack.color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, w, h);

        // Label above the box
        const fontSize = Math.max(11, Math.round(13 * (canvas.width / 640)));
        ctx.font = `bold ${fontSize}px monospace`;
        const label = `#${selectedTrack.id}`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = selectedTrack.color;
        ctx.fillRect(x1, y1 - fontSize - 4, tw + 6, fontSize + 4);
        ctx.fillStyle = "white";
        ctx.fillText(label, x1 + 3, y1 - 4);
      }
      // If no bbox for this camera, person is not detected here — leave canvas blank.
    });
  }, [selectedTrack, homographyData]);

  return (
    <div className="camera-grid">
      {LABELS.map((label, i) => (
        <div key={label} className="camera-cell">
          {images[i] ? (
            <img
              src={`data:image/jpeg;base64,${images[i]}`}
              alt={label}
              className="camera-img"
            />
          ) : (
            <div className="camera-placeholder">
              <span>{label}</span>
            </div>
          )}
          <span className="camera-label">{label}</span>
          <canvas
            ref={(el) => { canvasRefs.current[i] = el; }}
            className="camera-canvas"
          />
        </div>
      ))}
      {/* 8th cell — camera coverage button */}
      <div className="camera-cell coverage-cell" onClick={onShowCoverage}>
        <div className="coverage-cell-inner">
          <svg className="coverage-cell-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="3" width="20" height="18" rx="2" />
            <path d="M2 9h20M9 9v12" />
            <circle cx="16" cy="15" r="2.5" />
            <path d="M13.5 15h-4M16 12.5v-2" />
          </svg>
          <span className="coverage-cell-label">Camera Coverage</span>
          <span className="coverage-cell-hint">Click to view</span>
        </div>
      </div>
    </div>
  );
}
