import { useState, useEffect, useCallback } from "react";
import { monitoringApi, modelsApi } from "@/lib/api";
import type { Metrics, PerformancePoint, DriftReport, RegisteredModel, ExperimentRun } from "@/types";

export function useAdmin() {
    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [performance, setPerformance] = useState<PerformancePoint[]>([]);
    const [drift, setDrift] = useState<DriftReport | null>(null);
    const [models, setModels] = useState<RegisteredModel[]>([]);
    const [runs, setRuns] = useState<ExperimentRun[]>([]);
    const [loading, setLoading] = useState(true);
    const [windowDays, setWindowDays] = useState(30);

    const loadAll = useCallback(async () => {
        setLoading(true);
        try {
            const [m, p, d, reg, exp] = await Promise.allSettled([
                monitoringApi.getMetrics(),
                monitoringApi.getPerformance(windowDays),
                monitoringApi.getDrift(),
                modelsApi.getRegistry(),
                modelsApi.getExperiments("energy_demand_train"),
            ]);

            if (m.status === "fulfilled") setMetrics(m.value.data);
            if (p.status === "fulfilled") setPerformance(p.value.data);
            if (d.status === "fulfilled") setDrift(d.value.data);
            if (reg.status === "fulfilled") setModels(reg.value.data);
            if (exp.status === "fulfilled") setRuns(exp.value.data);
        } finally {
            setLoading(false);
        }
    }, [windowDays]);

    useEffect(() => {
        loadAll() // todo: this needs to be checked
    }, [loadAll]);

    return {
        metrics, performance, drift, models, runs,
        loading, windowDays, setWindowDays,
        reload: loadAll,
    };
}