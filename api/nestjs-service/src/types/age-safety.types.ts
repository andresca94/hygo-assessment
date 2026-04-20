export type Verdict = 'safe' | 'flagged' | 'uncertain';

export interface FaceResult {
  bbox: number[];
  face_confidence: number;
  estimated_age: number | null;
  age_interval: number[] | null;
  p_minor: number | null;
  domain_type: string;
  quality_flags: string[];
  verdict: Verdict;
  policy_reason: string;
}

export interface AgeSafetyResult {
  verdict: Verdict;
  risk_score: number;
  faces: FaceResult[];
  policy_reason: string;
  model_version: string;
}

export interface AgeSafetyBatchResult {
  items: AgeSafetyResult[];
  flagged_count: number;
  uncertain_count: number;
  safe_count: number;
}

export interface AgeSafetyHealth {
  status: string;
  model_ready: boolean;
  model_version: string;
  detector_backend: string;
  main_model_mode: string;
  aux_model_mode: string;
}
