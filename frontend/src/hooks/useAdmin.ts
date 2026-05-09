// import { useState, useEffect, useCallback } from "react";
// import { monitoringApi, modelsApi } from "@/lib/api";
// import type { Metrics, PerformancePoint, DriftReport, RegisteredModel, ExperimentRun } from "@/types";
//
// export function useAdmin() {
//     const [metrics, setMetrics] = useState<Metrics | null>(null);
//     const [performance, setPerformance] = useState<PerformancePoint[]>([]);
//     const [drift, setDrift] = useState<DriftReport | null>(null);
//     const [models, setModels] = useState<RegisteredModel[]>([]);
//     const [runs, setRuns] = useState<ExperimentRun[]>([]);
//     const [loading, setLoading] = useState(true);
//     const [windowDays, setWindowDays] = useState(30);
//
//     const loadAll = useCallback(async () => {
//         setLoading(true);
//         try {
//             const [m, p, d, reg, exp] = await Promise.allSettled([
//                 monitoringApi.getMetrics(),
//                 monitoringApi.getPerformance(windowDays),
//                 monitoringApi.getDrift(),
//                 modelsApi.getRegistry(),
//                 modelsApi.getExperiments("energy_demand_train"),
//             ]);
//
//             if (m.status === "fulfilled") setMetrics(m.value.data);
//             if (p.status === "fulfilled") setPerformance(p.value.data);
//             if (d.status === "fulfilled") setDrift(d.value.data);
//             if (reg.status === "fulfilled") setModels(reg.value.data);
//             if (exp.status === "fulfilled") setRuns(exp.value.data);
//         } finally {
//             setLoading(false);
//         }
//     }, [windowDays]);
//
//     useEffect(() => {
//         loadAll() // todo: this needs to be checked
//     }, [loadAll]);
//
//     return {
//         metrics, performance, drift, models, runs,
//         loading, windowDays, setWindowDays,
//         reload: loadAll,
//     };
// }

import {useState, useEffect, useCallback} from "react";
import {monitoringApi, modelsApi} from "@/lib/api";
import type {
    Metrics, PerformancePoint, DriftReport, RegisteredModel, ExperimentRun
} from "@/types";

export type ModelView = "multivariate" | "univariate";

export const EXPERIMENT_NAMES: Record<ModelView, string> = {
    multivariate: "energy_demand_train",
    univariate: "energy_demand_univariate_train",
};

interface AdminState {
    // Multivariate
    metrics: Metrics | null;
    performance: PerformancePoint[];
    // Univariate
    univariateMetrics: Metrics | null;
    univariatePerformance: PerformancePoint[];
    // Shared
    drift: DriftReport | null;
    models: RegisteredModel[];
    // Experiments per model
    runs: Record<ModelView, ExperimentRun[]>;
    // Loading states per section
    loadingMonitoring: boolean;
    loadingModels: boolean;
    loadingExperiments: boolean;
    loadingDrift: boolean;
}

export function useAdmin() {
    const [windowDays, setWindowDays] = useState(30);
    const [state, setState] = useState<AdminState>({
        metrics: null,
        performance: [],
        univariateMetrics: null,
        univariatePerformance: [],
        drift: null,
        models: [],
        runs: {multivariate: [], univariate: []},
        loadingMonitoring: true,
        loadingModels: true,
        loadingExperiments: true,
        loadingDrift: true,
    });

    // ── Load monitoring for both models ────────────────────────────────────────
    const loadMonitoring = useCallback(async (days: number) => {
        setState((s) => ({...s, loadingMonitoring: true}));
        const [m, p, um, up] = await Promise.all([
            monitoringApi.getMetrics(days).catch(() => null),
            monitoringApi.getPerformance(days).catch(() => null),
            monitoringApi.getUnivariateMetrics(days).catch(() => null),
            monitoringApi.getUnivariatePerformance(days).catch(() => null),
        ]);
        setState((s) => ({
            ...s,
            metrics: m?.data ?? s.metrics,
            performance: p?.data ?? s.performance,
            univariateMetrics: um?.data ?? s.univariateMetrics,
            univariatePerformance: up?.data ?? s.univariatePerformance,
            loadingMonitoring: false,
        }));
    }, []);

    // ── Load model registry ─────────────────────────────────────────────────────
    const loadModels = useCallback(async () => {
        setState((s) => ({...s, loadingModels: true}));
        const reg = await modelsApi.getRegistry().catch(() => null);
        setState((s) => ({
            ...s,
            models: reg?.data ?? s.models,
            loadingModels: false,
        }));
    }, []);

    // ── Load experiments for both models ────────────────────────────────────────
    const loadExperiments = useCallback(async () => {
        setState((s) => ({...s, loadingExperiments: true}));
        const [mv, uv] = await Promise.all([
            modelsApi.getExperiments(EXPERIMENT_NAMES.multivariate).catch(() => null),
            modelsApi.getExperiments(EXPERIMENT_NAMES.univariate).catch(() => null),
        ]);
        setState((s) => ({
            ...s,
            runs: {
                multivariate: mv?.data ?? s.runs.multivariate,
                univariate: uv?.data ?? s.runs.univariate,
            },
            loadingExperiments: false,
        }));
    }, []);

    // ── Load drift report ───────────────────────────────────────────────────────
    const loadDrift = useCallback(async () => {
        setState((s) => ({...s, loadingDrift: true}));
        const d = await monitoringApi.getDrift().catch(() => null);
        setState((s) => ({
            ...s,
            drift: d?.data ?? s.drift,
            loadingDrift: false,
        }));
    }, []);

    // ── Initial load — all sections in parallel ─────────────────────────────────
    useEffect(() => {
        loadMonitoring(windowDays);
        loadModels();
        loadExperiments();
        loadDrift();
    }, []);

    // ── Reload monitoring when window changes ───────────────────────────────────
    useEffect(() => {
        loadMonitoring(windowDays);
    }, [windowDays]);

    const reload = useCallback(() => {
        loadMonitoring(windowDays);
        loadModels();
        loadExperiments();
        loadDrift();
    }, [windowDays]);

    const isInitialLoading =
        state.loadingMonitoring &&
        state.loadingModels &&
        state.loadingExperiments &&
        state.loadingDrift;

    return {
        ...state,
        windowDays,
        setWindowDays,
        reload,
        isInitialLoading,
        reloadModels: loadModels,
    };
}