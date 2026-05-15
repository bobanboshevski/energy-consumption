import axios from "axios";
import type {
    ForecastPoint, HistoricalPoint, PerformancePoint,
    Metrics, DriftReport, RegisteredModel, ExperimentRun, UnivariateModelInfo, UnivariatePrediction,
    UnivariateRangePoint, BackendComparison
} from "@/types";

const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002",
    timeout: 240000,
});

export const predictionsApi = {
    getForecast: () => api.get<ForecastPoint[]>("/predictions/forecast"),
    getHistorical: (days = 90) => api.get<HistoricalPoint[]>(`/predictions/historical?days=${days}`),
    refreshData: () => api.post("/predictions/refresh"),
};

// export const monitoringApi = {
//     getMetrics: () => api.get<Metrics>("/monitoring/metrics"),
//     getPerformance: (windowDays = 30) => api.get<PerformancePoint[]>(`/monitoring/performance?window_days=${windowDays}`),
//     getDrift: () => api.get<DriftReport>("/monitoring/drift"),
//     getDriftReportUrl: () => `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002"}/monitoring/drift/report`,
// };

export const monitoringApi = {
    // Multivariate
    getMetrics: (windowDays = 30) =>
        api.get<Metrics>(`/monitoring/metrics?window_days=${windowDays}`),
    getPerformance: (windowDays = 30) =>
        api.get<PerformancePoint[]>(`/monitoring/performance?window_days=${windowDays}`),

    // Univariate
    getUnivariateMetrics: (windowDays = 30) =>
        api.get<Metrics>(`/monitoring/univariate/metrics?window_days=${windowDays}`),
    getUnivariatePerformance: (windowDays = 30) =>
        api.get<PerformancePoint[]>(`/monitoring/univariate/performance?window_days=${windowDays}`),

    // Drift (shared)
    getDrift: () => api.get<DriftReport>("/monitoring/drift"), // todo: i dont even use this
    getDriftReportUrl: () =>
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002"}/monitoring/drift/report`,

    getGx: () => api.get<DriftReport>("/monitoring/gx"),
    getGxReportUrl: () =>
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002"}/monitoring/gx/report`,

    getComparison: (windowDays = 30, modelKey: "multivariate" | "univariate" = "multivariate") =>
        api.get<BackendComparison>(
            `/monitoring/comparison?window_days=${windowDays}&model_key=${modelKey}`
        ),
};

export const modelsApi = {
    getRegistry: () => api.get<RegisteredModel[]>("/models/registry"),
    getVersions: (modelName: string) => api.get(`/models/versions/${modelName}`),
    getExperiments: (experimentName: string) =>
        api.get<ExperimentRun[]>(`/models/experiments/${experimentName}`),
    getActive: () => api.get("/models/active"),
    activate: (version: string, modelKey: "multivariate" | "univariate") =>
        api.post(`/models/activate?version=${version}&model_key=${modelKey}`),
};

export const univariateApi = {
    getInfo: () => api.get<UnivariateModelInfo>("/univariate/info"),
    predictDate: (targetDate: string) =>
        api.get<UnivariatePrediction>(`/univariate/predict?target_date=${targetDate}`),
    predictRange: (startDate: string, endDate: string) =>
        api.get<UnivariateRangePoint[]>(`/univariate/range?start_date=${startDate}&end_date=${endDate}`),
}