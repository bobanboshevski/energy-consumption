// export interface ForecastPoint {
//   date: string;
//   predicted_demand: number;
//   demand_category: "low" | "medium" | "high";
//   temp_max: number;
//   temp_min: number;
//   daylight_duration: number;
// }

export interface ForecastPoint {
    date: string;
    predicted_demand: number;
    demand_category: "low" | "medium" | "high";
    is_confirmed: boolean;
    temp_max: number | null;
    temp_min: number | null;
    daylight_duration: number | null;
}

export interface HistoricalPoint {
    Date: string;
    energy_demand: number;
    temp_max: number;
    temp_min: number;
    daylight_duration: number;
}

export interface PerformancePoint {
    date: string;
    actual: number;
    predicted: number;
    error: number;
}

export interface Metrics {
    mae: number;
    mse: number;
    rmse: number;
    mean_error: number;
    max_error: number;
    data_points: number;
}

export interface DriftReport {
    available: boolean;
    path?: string;
    size_kb?: number;
    last_modified?: number;
    reason?: string;
}

export interface ValidationReport {
    available: boolean;
    size_kb?: number;
    source?: string;
    passed?: boolean;
    reason?: string;
    error?: string;
}

export interface ModelVersion {
    version: string;
    stage: string;
    run_id: string;
    status: string;
}

export interface RegisteredModel {
    name: string;
    versions: ModelVersion[];
}

export interface ExperimentRun {
    run_id: string;
    run_name: string;
    status: string;
    start_time: number;
    metrics: Record<string, number>;
    params: Record<string, string>;
}

export type DemandCategory = "low" | "medium" | "high";

export interface UnivariatePrediction {
    target_date: string;
    predicted_demand: number;
    demand_category: DemandCategory;
    days_ahead: number;
    model: string;
    last_known_date: string;
    note: string;
}

export interface UnivariateRangePoint {
    date: string;
    predicted_demand: number;
    demand_category: DemandCategory;
    days_ahead: number;
}

export interface UnivariateModelInfo {
    model: string;
    description: string;
    features: string[];
    window_size: number;
    max_horizon_days: number;
    max_date: string;
    note: string;
}
