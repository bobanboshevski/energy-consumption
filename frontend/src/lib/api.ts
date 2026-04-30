// import axios from "axios";
//
// const api = axios.create({
//   baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002",
// });
//
// export const getForecast = () => api.get("/predictions/forecast");
// export const getHistorical = (days = 90) => api.get(`/predictions/historical?days=${days}`);
// export const getPerformance = (windowDays = 30) => api.get(`/monitoring/performance?window_days=${windowDays}`);
// export const getMetrics = () => api.get("/monitoring/metrics");
// export const getDrift = () => api.get("/monitoring/drift");
// export const getRegistry = () => api.get("/models/registry");
// export const getExperiments = (name: string) => api.get(`/models/experiments/${name}`);
// export const transitionModel = (model_name: string, version: string, stage: string) =>
//   api.post(`/models/transition?model_name=${model_name}&version=${version}&stage=${stage}`);

import axios from "axios";
import type {
  ForecastPoint, HistoricalPoint, PerformancePoint,
  Metrics, DriftReport, RegisteredModel, ExperimentRun
} from "@/types";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002",
  timeout: 30000,
});

export const predictionsApi = {
  getForecast: () => api.get<ForecastPoint[]>("/predictions/forecast"),
  getHistorical: (days = 90) => api.get<HistoricalPoint[]>(`/predictions/historical?days=${days}`),
  refreshData: () => api.post("/predictions/refresh"),
};

export const monitoringApi = {
  getMetrics: () => api.get<Metrics>("/monitoring/metrics"),
  getPerformance: (windowDays = 30) => api.get<PerformancePoint[]>(`/monitoring/performance?window_days=${windowDays}`),
  getDrift: () => api.get<DriftReport>("/monitoring/drift"),
  getDriftReportUrl: () => `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002"}/monitoring/drift/report`,
};

// todo: changes here
export const modelsApi = {
  getRegistry: () => api.get<RegisteredModel[]>("/models/registry"),
  getVersions: (modelName: string) => api.get(`/models/versions/${modelName}`),
  getExperiments: (name: string) => api.get<ExperimentRun[]>(`/models/experiments/${name}`),
  getActive: () => api.get("/models/active"),
  activate: (version: string) => api.post(`/models/activate?version=${version}`),
//   setAlias: (modelName: string, alias: string, version: string) =>
//     api.post(`/models/alias?model_name=${modelName}&alias=${alias}&version=${version}`),
//   removeAlias: (modelName: string, alias: string) =>
//     api.delete(`/models/alias?model_name=${modelName}&alias=${alias}`),
};