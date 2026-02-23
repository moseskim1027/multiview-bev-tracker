import type { Track } from "../types";

interface Props {
  tracks: Track[];
  selectedTrackId: number | null;
  onSelectTrack: (id: number) => void;
  onClearSelection: () => void;
}

export default function TracksPanel({
  tracks,
  selectedTrackId,
  onSelectTrack,
  onClearSelection,
}: Props) {
  return (
    <div className="tracks-panel">
      <div className="legend-header">
        <h3 className="legend-title">
          Tracks <span className="track-count">{tracks.length}</span>
        </h3>
        {selectedTrackId !== null && (
          <button className="btn-show-all" onClick={onClearSelection}>
            ✕ #{selectedTrackId}
          </button>
        )}
      </div>

      <div className="legend-list">
        {tracks.map((t) => (
          <div
            key={t.id}
            className={
              "legend-item" +
              (selectedTrackId === t.id ? " legend-item--selected" : "") +
              (selectedTrackId !== null && selectedTrackId !== t.id
                ? " legend-item--dimmed"
                : "")
            }
            onClick={() => onSelectTrack(t.id)}
            title={
              selectedTrackId === t.id
                ? `Tracking #${t.id} — click "✕" to reset`
                : `Highlight track #${t.id} across all cameras`
            }
          >
            <span className="legend-swatch" style={{ background: t.color }} />
            <span className="legend-id">#{t.id}</span>
            <span className="legend-pos">
              ({t.world_xy[0].toFixed(1)}, {t.world_xy[1].toFixed(1)}) m
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
