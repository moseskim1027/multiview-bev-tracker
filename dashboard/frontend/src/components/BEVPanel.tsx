import type { Track } from "../types";

interface Props {
  bevImage: string | undefined; // base64 JPEG
  tracks: Track[];
}

export default function BEVPanel({ bevImage, tracks }: Props) {
  return (
    <div className="bev-panel">
      <h2 className="bev-title">Bird's Eye View</h2>

      <div className="bev-img-wrap">
        {bevImage ? (
          <img
            src={`data:image/jpeg;base64,${bevImage}`}
            alt="BEV"
            className="bev-img"
          />
        ) : (
          <div className="bev-placeholder">
            <span>Waiting for data…</span>
          </div>
        )}
      </div>

      <div className="track-legend">
        <h3 className="legend-title">
          Active tracks <span className="track-count">{tracks.length}</span>
        </h3>
        <div className="legend-list">
          {tracks.map((t) => (
            <div key={t.id} className="legend-item">
              <span
                className="legend-swatch"
                style={{ background: t.color }}
              />
              <span className="legend-id">#{t.id}</span>
              <span className="legend-pos">
                ({t.world_xy[0].toFixed(1)}, {t.world_xy[1].toFixed(1)}) m
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
