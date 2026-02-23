export interface Track {
  id: number;
  world_xy: [number, number];
  history: [number, number][];
  color: string; // CSS hex e.g. "#ff3838"
  camera_bboxes: Record<number, [number, number, number, number]>; // cam_idx → [x1,y1,x2,y2]
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

export interface HomographyData {
  homographies: number[][][]; // [7][3][3] — world metres → full-res image pixels
  orig_w: number;
  orig_h: number;
}
