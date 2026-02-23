import { useEffect } from "react";

interface LegendItem {
  cam: string;
  color: string;
}

interface Props {
  image: string; // base64 JPEG
  legend: LegendItem[];
  onClose: () => void;
}

export default function CoverageModal({ image, legend, onClose }: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="coverage-backdrop" onClick={onClose}>
      <div className="coverage-modal" onClick={(e) => e.stopPropagation()}>
        <div className="coverage-modal-header">
          <span className="coverage-modal-title">Camera Coverage Map</span>
          <button className="coverage-modal-close" onClick={onClose}>✕</button>
        </div>
        <img
          src={`data:image/jpeg;base64,${image}`}
          alt="Camera coverage map"
          className="coverage-modal-img"
        />
        <div className="coverage-legend">
          {legend.map((item) => (
            <div key={item.cam} className="coverage-legend-item">
              <span className="coverage-legend-swatch" style={{ background: item.color }} />
              <span className="coverage-legend-label">{item.cam}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
