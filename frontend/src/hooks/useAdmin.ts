import {useReducer, useState, useEffect, useCallback} from "react";
import {monitoringApi, modelsApi} from "@/lib/api";
import type {
    Metrics, PerformancePoint, DriftReport, RegisteredModel,
    ExperimentRun, ValidationReport, BackendComparison,
} from "@/types";

export type ModelView = "multivariate" | "univariate";

export const EXPERIMENT_NAMES: Record<ModelView, string> = {
    multivariate: "energy_demand_train",
    univariate: "energy_demand_univariate_train",
};

// ── State ─────────────────────────────────────────────────────────────────────

interface AdminState {
    metrics: Metrics | null;
    performance: PerformancePoint[];
    univariateMetrics: Metrics | null;
    univariatePerformance: PerformancePoint[];
    drift: DriftReport | null;
    validationReport: ValidationReport | null;
    models: RegisteredModel[];
    runs: Record<ModelView, ExperimentRun[]>;
    comparisonMultivariate: BackendComparison | null;
    comparisonUnivariate: BackendComparison | null;
    loadingMonitoring: boolean;
    loadingModels: boolean;
    loadingExperiments: boolean;
    loadingDrift: boolean;
    loadingValidation: boolean;
    loadingComparison: boolean;
}

const INITIAL_STATE: AdminState = {
    metrics: null,
    performance: [],
    univariateMetrics: null,
    univariatePerformance: [],
    drift: null,
    validationReport: null,
    models: [],
    runs: {multivariate: [], univariate: []},
    comparisonMultivariate: null,
    comparisonUnivariate: null,
    loadingMonitoring: true,
    loadingModels: true,
    loadingExperiments: true,
    loadingDrift: true,
    loadingValidation: true,
    loadingComparison: true,
};

// ── Actions ───────────────────────────────────────────────────────────────────

type AdminAction =
    | { type: "MONITORING_START" }
    | {
    type: "MONITORING_DONE";
    metrics: Metrics | null;
    performance: PerformancePoint[];
    univariateMetrics: Metrics | null;
    univariatePerformance: PerformancePoint[];
}
    | { type: "MODELS_START" }
    | { type: "MODELS_DONE"; models: RegisteredModel[] }
    | { type: "EXPERIMENTS_START" }
    | { type: "EXPERIMENTS_DONE"; runs: Record<ModelView, ExperimentRun[]> }
    | { type: "DRIFT_START" }
    | { type: "DRIFT_DONE"; drift: DriftReport | null }
    | { type: "VALIDATION_START" }
    | { type: "VALIDATION_DONE"; validationReport: ValidationReport | null }
    | { type: "COMPARISON_START" }
    | {
    type: "COMPARISON_DONE";
    comparisonMultivariate: BackendComparison | null;
    comparisonUnivariate: BackendComparison | null;
};

// ── Reducer ───────────────────────────────────────────────────────────────────

function adminReducer(state: AdminState, action: AdminAction): AdminState {
    switch (action.type) {
        case "MONITORING_START":
            return {...state, loadingMonitoring: true};
        case "MONITORING_DONE":
            return {
                ...state,
                loadingMonitoring: false,
                metrics: action.metrics ?? state.metrics,
                performance: action.performance,
                univariateMetrics: action.univariateMetrics ?? state.univariateMetrics,
                univariatePerformance: action.univariatePerformance,
            };
        case "MODELS_START":
            return {...state, loadingModels: true};
        case "MODELS_DONE":
            return {...state, loadingModels: false, models: action.models};
        case "EXPERIMENTS_START":
            return {...state, loadingExperiments: true};
        case "EXPERIMENTS_DONE":
            return {...state, loadingExperiments: false, runs: action.runs};
        case "DRIFT_START":
            return {...state, loadingDrift: true};
        case "DRIFT_DONE":
            return {...state, loadingDrift: false, drift: action.drift};
        case "VALIDATION_START":
            return {...state, loadingValidation: true};
        case "VALIDATION_DONE":
            return {...state, loadingValidation: false, validationReport: action.validationReport};
        case "COMPARISON_START":
            return {...state, loadingComparison: true};
        case "COMPARISON_DONE":
            return {
                ...state,
                loadingComparison: false,
                comparisonMultivariate: action.comparisonMultivariate ?? state.comparisonMultivariate,
                comparisonUnivariate: action.comparisonUnivariate ?? state.comparisonUnivariate,
            };
    }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useAdmin() {
    const [windowDays, setWindowDays] = useState(30);
    const [state, dispatch] = useReducer(adminReducer, INITIAL_STATE);

    // All loaders close over only `dispatch` which is stable — [] deps are correct.
    // dispatch from useReducer is guaranteed stable by React (same guarantee as setState).

    const loadMonitoring = useCallback(async (days: number) => {
        dispatch({type: "MONITORING_START"});
        const [m, p, um, up] = await Promise.all([
            monitoringApi.getMetrics(days).catch(() => null),
            monitoringApi.getPerformance(days).catch(() => null),
            monitoringApi.getUnivariateMetrics(days).catch(() => null),
            monitoringApi.getUnivariatePerformance(days).catch(() => null),
        ]);
        dispatch({
            type: "MONITORING_DONE",
            metrics: m?.data ?? null,
            performance: p?.data ?? [],
            univariateMetrics: um?.data ?? null,
            univariatePerformance: up?.data ?? [],
        });
    }, []);

    const loadModels = useCallback(async () => {
        dispatch({type: "MODELS_START"});
        const reg = await modelsApi.getRegistry().catch(() => null);
        dispatch({type: "MODELS_DONE", models: reg?.data ?? []});
    }, []);

    const loadExperiments = useCallback(async () => {
        dispatch({type: "EXPERIMENTS_START"});
        const [mv, uv] = await Promise.all([
            modelsApi.getExperiments(EXPERIMENT_NAMES.multivariate).catch(() => null),
            modelsApi.getExperiments(EXPERIMENT_NAMES.univariate).catch(() => null),
        ]);
        dispatch({
            type: "EXPERIMENTS_DONE",
            runs: {
                multivariate: mv?.data ?? [],
                univariate: uv?.data ?? [],
            },
        });
    }, []);

    const loadDrift = useCallback(async () => {
        dispatch({type: "DRIFT_START"});
        const d = await monitoringApi.getDrift().catch(() => null);
        dispatch({type: "DRIFT_DONE", drift: d?.data ?? null});
    }, []);

    const loadValidation = useCallback(async () => {
        dispatch({type: "VALIDATION_START"});
        const v = await monitoringApi.getGx().catch(() => null);
        dispatch({type: "VALIDATION_DONE", validationReport: v?.data ?? null});
    }, []);

    const loadComparison = useCallback(async (days: number) => {
        dispatch({type: "COMPARISON_START"});
        const [mv, uv] = await Promise.all([
            monitoringApi.getComparison(days, "multivariate").catch(() => null),
            monitoringApi.getComparison(days, "univariate").catch(() => null),
        ]);
        dispatch({
            type: "COMPARISON_DONE",
            comparisonMultivariate: mv?.data ?? null,
            comparisonUnivariate: uv?.data ?? null,
        });
    }, []);

    // ── Effect 1: stable data — runs once (all deps are [] callbacks) ──────────
    useEffect(() => {
        void loadModels();
        void loadExperiments();
        void loadDrift();
        void loadValidation();
    }, [loadModels, loadExperiments, loadDrift, loadValidation]);

    // ── Effect 2: window-dependent data — re-runs when windowDays changes ──────
    useEffect(() => {
        void loadMonitoring(windowDays);
        void loadComparison(windowDays);
    }, [windowDays, loadMonitoring, loadComparison]);

    // ── Manual full reload ─────────────────────────────────────────────────────
    const reload = useCallback(() => {
        void loadModels();
        void loadExperiments();
        void loadDrift();
        void loadValidation();
        void loadMonitoring(windowDays);
        void loadComparison(windowDays);
    }, [windowDays, loadModels, loadExperiments, loadDrift, loadValidation, loadMonitoring, loadComparison]);

    const isInitialLoading =
        state.loadingMonitoring &&
        state.loadingModels &&
        state.loadingExperiments &&
        state.loadingDrift &&
        state.loadingValidation;

    return {
        ...state,
        windowDays,
        setWindowDays,
        reload,
        isInitialLoading,
        reloadModels: loadModels,
    };
}