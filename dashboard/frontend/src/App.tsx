import { useCallback, useEffect, useRef, useState } from "react";
import BEVPanel from "./components/BEVPanel";
import CameraGrid from "./components/CameraGrid";
import type { FrameData, InfoData } from "./types";

const FPS = 2; // playback speed (frames per second)

export default function App() {
  const [info, setInfo] = useState<InfoData | null>(null);
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  const playingRef = useRef(playing);
  playingRef.current = playing;

  // ── Poll /api/info every 2 s ───────────────────────────────────────────────
  useEffect(() => {
    const poll = () =>
      fetch("/api/info")
        .then((r) => r.json())
        .then((d: InfoData) => setInfo(d))
        .catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  // ── Auto-start playback once frame 0 is ready ─────────────────────────────
  useEffect(() => {
    if (info && info.processed_up_to >= 0 && !playingRef.current) {
      setPlaying(true);
    }
  }, [info?.processed_up_to]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Fetch frame data whenever frameIdx or selection changes ──────────────
  useEffect(() => {
    if (!info || frameIdx > info.processed_up_to) return;
    const controller = new AbortController();
    const url =
      selectedTrackId !== null
        ? `/api/frame/${frameIdx}?selected=${selectedTrackId}`
        : `/api/frame/${frameIdx}`;
    fetch(url, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((d: FrameData | null) => {
        if (d) setFrameData(d);
      })
      .catch(() => {});
    return () => controller.abort();
  }, [frameIdx, info?.processed_up_to, selectedTrackId]);

  // ── Playback interval ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setFrameIdx((prev) => {
        if (!info) return prev;
        const next = prev + 1;
        // If the next frame isn't processed yet, stay put
        if (next > info.processed_up_to) return prev;
        // Loop back to 0 when we've seen everything
        if (next >= info.num_frames) {
          setPlaying(false);
          return prev;
        }
        return next;
      });
    }, 1000 / FPS);
    return () => clearInterval(id);
  }, [playing, info]);

  // ── Slider seek ───────────────────────────────────────────────────────────
  const handleSeek = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = Number(e.target.value);
      setFrameIdx(v);
      setPlaying(false);
    },
    []
  );

  // ── Track selection ───────────────────────────────────────────────────────
  const handleSelectTrack = useCallback((id: number) => {
    setSelectedTrackId((prev) => (prev === id ? null : id));
  }, []);

  const maxFrame = info ? info.processed_up_to : 0;
  const totalFrames = info ? info.num_frames : 0;
  const processedPct = totalFrames > 0 ? Math.round((maxFrame / totalFrames) * 100) : 0;

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="header">
        <h1 className="header-title">BEV Tracker</h1>

        <div className="controls">
          <button
            className="btn-play"
            onClick={() => setPlaying((p) => !p)}
            disabled={maxFrame < 0}
          >
            {playing ? "⏸ Pause" : "▶ Play"}
          </button>

          <input
            type="range"
            min={0}
            max={Math.max(maxFrame, 0)}
            value={frameIdx}
            onChange={handleSeek}
            className="seek-slider"
          />

          <span className="frame-counter">
            Frame {frameIdx} / {maxFrame}
          </span>
        </div>

        <div className="status">
          {!info?.is_ready && <span className="badge badge--loading">Loading models…</span>}
          {info?.is_ready && maxFrame < totalFrames - 1 && (
            <span className="badge badge--processing">
              Processing {processedPct}%
            </span>
          )}
          {info?.is_ready && maxFrame >= totalFrames - 1 && (
            <span className="badge badge--done">Ready</span>
          )}
          {frameData && (
            <span className="badge badge--tracks">
              {frameData.tracks.length} tracks
            </span>
          )}
        </div>
      </header>

      {/* ── Main layout ────────────────────────────────────────────────────── */}
      <main className="main">
        <section className="cameras-section">
          <CameraGrid images={frameData?.camera_images ?? []} />
        </section>

        <aside className="bev-section">
          <BEVPanel
            bevImage={frameData?.bev_image}
            tracks={frameData?.tracks ?? []}
            selectedTrackId={selectedTrackId}
            onSelectTrack={handleSelectTrack}
          />
        </aside>
      </main>
    </div>
  );
}
