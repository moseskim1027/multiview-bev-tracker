interface Props {
  images: string[]; // base64 JPEG, indices 0-6 correspond to C1-C7
}

const LABELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];

export default function CameraGrid({ images }: Props) {
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
        </div>
      ))}
      {/* 8th cell — empty filler to complete the 4×2 grid */}
      <div className="camera-cell camera-cell--empty" />
    </div>
  );
}
