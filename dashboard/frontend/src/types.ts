export interface Track {
  id: number;
  world_xy: [number, number];
  history: [number, number][];
  color: string; // CSS hex e.g. "#ff3838"
}

export interface FrameData {
  frame_idx: number;
  tracks: Track[];
  camera_images: string[]; // base64 JPEG, one per camera
  bev_image: string; // base64 JPEG
}

export interface InfoData {
  num_frames: number;
  num_cameras: number;
  processed_up_to: number;
  is_ready: boolean;
}
