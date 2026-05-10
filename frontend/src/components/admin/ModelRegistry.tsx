"use client";

import {useState, useEffect, useCallback} from "react";
import type {RegisteredModel} from "@/types";
import type {ModelView} from "@/hooks/useAdmin";
import {InlineSpinner} from "@/components/ui/LoadingSpinner";
import {VersionsModal} from "@/components/admin/VersionsModal";
import {modelsApi} from "@/lib/api";

interface VersionDetail {
    version: string;
    stage: string;
    run_id: string;
    status: string;
    creation_timestamp: number;
    metrics: Record<string, number>;
    params: Record<string, string>;
}

interface ActiveState {
    multivariate: { active_version: string; loaded_version: string; model_name: string };
    univariate: { active_version: string; loaded_version: string; model_name: string };
}

interface Props {
    models: RegisteredModel[];
    loadingModels: boolean;
    onReload?: () => void;
}

const MODEL_KEY_MAP: Record<string, ModelView> = {
    energy_demand_model: "multivariate",
    energy_demand_univariate_model: "univariate",
};

const MODEL_LABELS: Record<ModelView, string> = {
    multivariate: "Multivariate — weather + energy demand · 16-day forecast",
    univariate: "Univariate — energy demand only · 365-day forecast",
};

const MODEL_ICONS: Record<ModelView, string> = {
    multivariate: "🌤",
    univariate: "📈",
};

export function ModelRegistry({models, loadingModels, onReload}: Props) {
    const [activeState, setActiveState] = useState<ActiveState | null>(null);
    const [loadingActive, setLoadingActive] = useState(true);

    // Per-model version details — fetched on demand
    const [details, setDetails] = useState<Record<string, VersionDetail[]>>({});
    const [loadingVersions, setLoadingVersions] = useState<Record<string, boolean>>({});

    // Modal state
    const [modalModel, setModalModel] = useState<string | null>(null);

    // Activating state
    const [activating, setActivating] = useState<string | null>(null);

    const loadActiveState = useCallback(async () => {
        setLoadingActive(true);
        try {
            const r = await modelsApi.getActive();
            setActiveState(r.data);
        } finally {
            setLoadingActive(false);
        }
    }, []);

    useEffect(() => {
        loadActiveState();
    }, [loadActiveState]);

    const loadVersions = useCallback(async (modelName: string) => {
        if (details[modelName]) return; // already loaded
        setLoadingVersions((prev) => ({...prev, [modelName]: true}));
        try {
            const res = await modelsApi.getVersions(modelName);
            setDetails((prev) => ({...prev, [modelName]: res.data}));
        } catch (e) {
            console.error(`Failed to load versions for ${modelName}`, e);
        } finally {
            setLoadingVersions((prev) => ({...prev, [modelName]: false}));
        }
    }, [details]);

    const handleOpenModal = (modelName: string) => {
        setModalModel(modelName);
        loadVersions(modelName);
    };

    const handleCloseModal = () => setModalModel(null);

    const getActiveVersionForModel = (modelName: string) => {

        if (!activeState) return {active: "latest", loaded: null, isLoaded: false};
        const key = MODEL_KEY_MAP[modelName] ?? "multivariate";
        return {
            active: activeState[key]?.active_version ?? "latest",
            loaded: activeState[key]?.loaded_version ?? null,
            isLoaded: activeState[key]?.is_loaded ?? false,
        };

        // if (!activeState) return {active: "latest", loaded: "unknown"};
        // const key = MODEL_KEY_MAP[modelName] ?? "multivariate";
        // return {
        //     active: activeState[key]?.active_version ?? "latest",
        //     loaded: activeState[key]?.loaded_version ?? "unknown",
        // };
    };

    const handleActivate = async (modelName: string, version: string) => {
        const modelKey = MODEL_KEY_MAP[modelName] ?? "multivariate";
        setActivating(version);
        try {
            await modelsApi.activate(version, modelKey);
            await loadActiveState();
            onReload?.();
        } finally {
            setActivating(null);
        }
    };

    const handleResetToLatest = async (modelName: string) => {
        const modelKey = MODEL_KEY_MAP[modelName] ?? "multivariate";
        setActivating("latest");
        try {
            await modelsApi.activate("latest", modelKey);
            await loadActiveState();
        } finally {
            setActivating(null);
        }
    };

    if (loadingModels) {
        return (
            <div className="flex items-center justify-center py-16">
                <div className="flex flex-col items-center gap-3">
                    <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"/>
                    <p className="text-sm text-gray-400">Loading model registry...</p>
                </div>
            </div>
        );
    }

    if (models.length === 0) {
        return <p className="text-gray-500 text-sm py-8 text-center">No registered models found.</p>;
    }

    return (
        <div className="space-y-4">
            {models.map((m) => {
                const modelKey = MODEL_KEY_MAP[m.name] ?? "multivariate";
                const {active, loaded, isLoaded} = getActiveVersionForModel(m.name);
                const isResettingLatest = activating === "latest";

                return (
                    <div key={m.name} className="bg-gray-800/40 border border-gray-700 rounded-2xl overflow-hidden">
                        {/* ── Model row ─────────────────────────────────────────────────── */}
                        <div className="flex items-center gap-4 px-5 py-4">
                            {/* Icon */}
                            <div
                                className="w-10 h-10 bg-gray-900 border border-gray-700 rounded-xl flex items-center justify-center text-lg shrink-0">
                                {MODEL_ICONS[modelKey]}
                            </div>

                            {/* Name + label */}
                            <div className="flex-1 min-w-0">
                                <p className="font-semibold text-white truncate">{m.name}</p>
                                <p className="text-xs text-gray-500 mt-0.5 truncate">{MODEL_LABELS[modelKey]}</p>
                            </div>

                            {/* Active / loaded */}
                            {loadingActive ? (
                                <div className="flex items-center gap-1.5 text-xs text-gray-500">
                                    <InlineSpinner/> Loading...
                                </div>
                            ) : (
                                <div className="text-right shrink-0">
                                    <div className="flex items-center gap-2 justify-end mb-1">
                                        <span className="text-xs text-gray-500">Activated:</span>
                                        <span className="font-mono text-xs text-white bg-gray-900 px-2 py-0.5 rounded">
                                            v{active}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2 justify-end">
                                        <span className="text-xs text-gray-500">Loaded:</span>
                                        {isLoaded ? (
                                            <span
                                                className="font-mono text-xs text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded">
                                                v{loaded}
                                            </span>
                                        ) : (
                                            <span
                                                className="text-xs text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded"
                                                title="Will load on next prediction request"
                                            >
                                                pending
                                            </span>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Reset to latest */}
                            <button
                                onClick={() => handleResetToLatest(m.name)}
                                disabled={active === "latest" || !!activating}
                                className="text-xs px-3 py-1.5 border border-gray-600 text-gray-400 rounded-lg hover:border-blue-600 hover:text-blue-400 disabled:opacity-40 transition-colors flex items-center gap-1.5 shrink-0"
                            >
                                {isResettingLatest ? <InlineSpinner/> : null}
                                Use latest
                            </button>

                            {/* View versions button */}
                            <button
                                onClick={() => handleOpenModal(m.name)}
                                className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors shrink-0"
                            >
                                Versions
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                          d="M9 5l7 7-7 7"/>
                                </svg>
                            </button>
                        </div>

                        {/* ── Quick stats bar ────────────────────────────────────────────── */}
                        <div className="border-t border-gray-700/50 px-5 py-2.5 flex items-center gap-6 bg-gray-900/30">
              <span className="text-xs text-gray-500">
                {m.versions.length} version{m.versions.length !== 1 ? "s" : ""} registered
              </span>
                            {m.versions[0] && (
                                <span className="text-xs text-gray-500">
                  Latest: <span className="text-white font-mono">v{m.versions[0].version}</span>
                </span>
                            )}
                        </div>
                    </div>
                );
            })}

            {/* ── Modal ─────────────────────────────────────────────────────────── */}
            {modalModel && (() => {
                const modelKey = MODEL_KEY_MAP[modalModel] ?? "multivariate";
                const {active} = getActiveVersionForModel(modalModel);
                return (
                    <VersionsModal
                        open={!!modalModel}
                        onClose={handleCloseModal}
                        modelName={modalModel}
                        modelLabel={MODEL_LABELS[modelKey]}
                        versions={details[modalModel] ?? []}
                        loading={loadingVersions[modalModel] ?? false}
                        activeVersion={active}
                        activating={activating}
                        onActivate={(version) => handleActivate(modalModel, version)}
                    />
                );
            })()}
        </div>
    );
}