import { useCallback, useEffect, useRef, useState } from "react";
import BEVPanel from "./components/BEVPanel";
import CameraGrid from "./components/CameraGrid";
import CoverageModal from "./components/CoverageModal";
import TracksPanel from "./components/TracksPanel";
import type { FrameData, HomographyData, InfoData } from "./types";

const FPS = 2;

export default function App() {
  const [info, setInfo] = useState<InfoData | null>(null);
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  // True only once the current frameData was fetched with the active selection,
  // so the canvas overlay only draws when the plain camera images are loaded.
  const [selectionReady, setSelectionReady] = useState(false);
  const [homographyData, setHomographyData] = useState<HomographyData | null>(null);
  const [coverageData, setCoverageData] = useState<{ image: string; legend: { cam: string; color: string }[] } | null>(null);
  const [showCoverage, setShowCoverage] = useState(false);

  const playingRef = useRef(playing);
  playingRef.current = playing;

  // ── Fetch static data — retry once is_ready to handle 503 during model load
  useEffect(() => {
    fetch("/api/homographies")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: HomographyData | null) => { if (d) setHomographyData(d); })
      .catch(() => {});
    fetch("/api/bev-coverage")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { image: string; legend: { cam: string; color: string }[] } | null) => { if (d) setCoverageData(d); })
      .catch(() => {});
  }, []);

  // Retry coverage fetch once backend is ready (handles 503 during model load)
  useEffect(() => {
    if (!info?.is_ready || coverageData) return;
    fetch("/api/bev-coverage")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { image: string; legend: { cam: string; color: string }[] } | null) => { if (d) setCoverageData(d); })
      .catch(() => {});
  }, [info?.is_ready, coverageData]);

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

  // ── Fetch frame data whenever frameIdx or selection changes ───────────────
  useEffect(() => {
    if (!info || frameIdx > info.processed_up_to) return;
    const controller = new AbortController();
    const url =
      selectedTrackId !== null
        ? `/api/frame/${frameIdx}?selected=${selectedTrackId}`
        : `/api/frame/${frameIdx}`;
    setSelectionReady(false); // clear until the new plain-image frame arrives
    fetch(url, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((d: FrameData | null) => {
        if (d) {
          setFrameData(d);
          setSelectionReady(selectedTrackId !== null);
        }
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
        if (next > info.processed_up_to) return prev;
        if (next >= info.num_frames) { setPlaying(false); return prev; }
        return next;
      });
    }, 1000 / FPS);
    return () => clearInterval(id);
  }, [playing, info]);

  // ── Slider seek ───────────────────────────────────────────────────────────
  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setFrameIdx(Number(e.target.value));
    setPlaying(false);
  }, []);

  // ── Track selection ───────────────────────────────────────────────────────
  const handleSelectTrack = useCallback((id: number) => {
    setSelectedTrackId((prev) => (prev === id ? null : id));
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedTrackId(null);
    setSelectionReady(false);
  }, []);

  const maxFrame = info ? info.processed_up_to : 0;
  const totalFrames = info ? info.num_frames : 0;
  const processedPct = totalFrames > 0 ? Math.round((maxFrame / totalFrames) * 100) : 0;

  return (
    <div className="app">
      {showCoverage && coverageData && (
        <CoverageModal image={coverageData.image} legend={coverageData.legend} onClose={() => setShowCoverage(false)} />
      )}

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
          {!info?.is_ready && (
            <span className="badge badge--loading">Loading models…</span>
          )}
          {info?.is_ready && maxFrame < totalFrames - 1 && (
            <span className="badge badge--processing">Processing {processedPct}%</span>
          )}
          {info?.is_ready && maxFrame >= totalFrames - 1 && (
            <span className="badge badge--done">Ready</span>
          )}
          {frameData && (
            <span className="badge badge--tracks">{frameData.tracks.length} tracks</span>
          )}
        </div>
      </header>

      {/* ── Main: left column (BEV + cameras) + right panel (tracks) ─────── */}
      <main className="main">

        <div className="content-left">
          {/* BEV — horizontal, top of left column */}
          <section className="bev-section">
            <div className="bev-section-label">Bird's Eye View</div>
            <BEVPanel bevImage={frameData?.bev_image} />
          </section>

          {/* Camera grid — below BEV */}
          <section className="cameras-section">
            <CameraGrid
              images={frameData?.camera_images ?? []}
              selectedTrack={selectionReady ? frameData?.tracks.find((t) => t.id === selectedTrackId) : undefined}
              homographyData={homographyData}
              onShowCoverage={() => setShowCoverage(true)}
            />
          </section>
        </div>

        {/* Tracks — right panel */}
        <TracksPanel
          tracks={frameData?.tracks ?? []}
          selectedTrackId={selectedTrackId}
          onSelectTrack={handleSelectTrack}
          onClearSelection={handleClearSelection}
        />

      </main>
    </div>
  );
}
