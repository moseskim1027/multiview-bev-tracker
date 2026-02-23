interface Props {
  bevImage: string | undefined; // base64 JPEG (landscape)
}

export default function BEVPanel({ bevImage }: Props) {
  return (
    <div className="bev-view">
      {bevImage ? (
        <img
          src={`data:image/jpeg;base64,${bevImage}`}
          alt="Bird's Eye View"
          className="bev-img"
        />
      ) : (
        <div className="bev-placeholder">
          <span>Waiting for data…</span>
        </div>
      )}
    </div>
  );
}
