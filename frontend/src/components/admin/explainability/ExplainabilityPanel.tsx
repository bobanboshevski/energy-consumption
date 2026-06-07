"use client";
import {useEffect, useState} from "react";
import {useExplainability} from "@/hooks/useExplainability";
import {DateSelector} from "./DateSelector";
import {VariantSelector} from "./VariantSelector";
import {FeatureImportanceChart} from "./FeatureImportanceChart";
import {TimestepChart} from "./TimestepChart";
import {ShapHeatmap} from "./ShapHeatmap";
import {ALL_VARIANTS, VARIANT_CONFIG} from "./constants";
import type {ShapArtifact, ShapVariant} from "@/types/explainability";
import {LoadingSpinner} from "@/components/ui/LoadingSpinner";
import {useShapNarratives} from "@/hooks/useShapNarratives";
import {NarrativeSummary} from "@/components/admin/explainability/NarrativeSummary";

interface Props {
    /** If provided, loads explanations for a specific version. Otherwise uses active. */
    version?: string;
}

export function ExplainabilityPanel({version}: Props) {
    // ── All hooks before any early returns ────────────────────────────────────
    const {keras, onnx, onnxQuantized, loading, errors, load} = useExplainability();
    const {get, fetchFor} = useShapNarratives(version);

    useEffect(() => {
        void load(version);
    }, [load, version]);

    // manualDate: the date the admin explicitly selected.
    // null means "auto-select the first available date from loaded data".
    const [manualDate, setManualDate] = useState<string | null>(null);

    // visible: which variants are toggled on by the admin
    const [visible, setVisible] = useState<Set<ShapVariant>>(new Set(ALL_VARIANTS));

    const anyArtifact = keras ?? onnx ?? onnxQuantized;
    const availableDates = anyArtifact?.explanations.map((e) => e.date) ?? [];
    const firstDate = availableDates[0] ?? null;

    // effectiveDate: user's choice when still valid in the loaded data,
    // otherwise falls back to firstDate automatically.
    const effectiveDate =
        manualDate && availableDates.includes(manualDate) ? manualDate : firstDate;

    // available: variants that loaded successfully
    const available = new Set<ShapVariant>(
        ALL_VARIANTS.filter((v) => {
            if (v === "keras") return keras !== null;
            if (v === "onnx") return onnx !== null;
            return onnxQuantized !== null;
        })
    );

    // artifactMap: null values stripped — chart components expect ShapArtifact, not null
    const artifactMap: Partial<Record<ShapVariant, ShapArtifact>> = {
        ...(keras && {keras}),
        ...(onnx && {onnx}),
        ...(onnxQuantized && {onnx_quantized: onnxQuantized}),
    };

    // visibleAndAvailable: toggled-on AND successfully loaded.
    const visibleAndAvailable = ALL_VARIANTS.filter(
        (v) => visible.has(v) && available.has(v)
    );

    const visibleKey = visibleAndAvailable.join(",");
    useEffect(() => {
        if (effectiveDate && visibleAndAvailable.length > 0) {
            fetchFor(effectiveDate, visibleAndAvailable);
        }
        // visibleKey is a stable proxy for the visibleAndAvailable array identity
    }, [effectiveDate, visibleKey, fetchFor]);

    // ── Early returns (all hooks already called above) ────────────────────────
    if (loading) {
        return <LoadingSpinner text="Loading SHAP explanations..."/>;
    }

    if (available.size === 0) {
        const firstError = errors.keras ?? errors.onnx ?? errors.onnx_quantized;
        return (
            <div className="rounded-xl border border-amber-800/50 bg-amber-950/20 p-5">
                <p className="text-sm font-semibold text-amber-400 mb-1">Explanations unavailable</p>
                <p className="text-xs text-gray-400">{firstError}</p>
            </div>
        );
    }

    if (!effectiveDate || !anyArtifact) return null;

    // ── Component logic ───────────────────────────────────────────────────────
    const failedVariants = ALL_VARIANTS.filter((v) => !available.has(v));

    const toggleVariant = (v: ShapVariant) => {
        setVisible((prev) => {
            const next = new Set(prev);
            next.has(v) ? next.delete(v) : next.add(v);
            return next;
        });
    };

    const explanationFor = (v: ShapVariant) => {
        const art = v === "keras" ? keras : v === "onnx" ? onnx : onnxQuantized;
        return art?.explanations.find((e) => e.date === effectiveDate) ?? null;
    };

    const windowSize = anyArtifact.window_size;
    const featureNames = anyArtifact.feature_names;
    const modelVersion = anyArtifact.version;

    return (
        <div className="space-y-8">
            {/* ── Header ──────────────────────────────────────────────────────── */}
            <div className="flex items-center justify-between flex-wrap gap-3">
                <div className="text-xs text-gray-500 space-x-3">
                    {modelVersion && (
                        <span>
                            Version: <span className="font-mono text-gray-300">v{modelVersion}</span>
                        </span>
                    )}
                    <span>
                        Background samples:{" "}
                        <span className="font-mono text-gray-300">{anyArtifact.n_background_samples}</span>
                    </span>
                    <span>
                        Window size:{" "}
                        <span className="font-mono text-gray-300">{windowSize} days</span>
                    </span>
                </div>

                {/* Error badges for partially failed variants */}
                {failedVariants.length > 0 && (
                    <div className="flex gap-2 flex-wrap">
                        {failedVariants.map((v) => (
                            <span
                                key={v}
                                className="text-xs px-2 py-1 rounded-lg bg-red-950/30 border border-red-800/40 text-red-400"
                                title={errors[v] ?? ""}
                            >
                                {VARIANT_CONFIG[v].label}: unavailable
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Controls ─────────────────────────────────────────────────────── */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4 bg-gray-900/40 border border-gray-800 rounded-xl">
                <DateSelector
                    explanations={anyArtifact.explanations}
                    selected={effectiveDate}
                    onChange={setManualDate}
                />
                <VariantSelector
                    visible={visible}
                    available={available}
                    onChange={toggleVariant}
                />
            </div>

            {/* ── Demand summary cards ──────────────────────────────────────────── */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {visibleAndAvailable.map((v) => {
                    const exp = explanationFor(v);
                    if (!exp) return null;
                    const cfg = VARIANT_CONFIG[v];
                    return (
                        <div key={v} className={`rounded-xl border p-4 ${cfg.badgeBg} ${cfg.badgeBorder}`}>
                            <p className={`text-xs font-semibold mb-1 ${cfg.badgeText}`}>{cfg.label}</p>
                            <p className="text-2xl font-bold font-mono text-white">
                                {exp.predicted_demand.toFixed(3)}
                                <span className="text-sm text-gray-400 ml-1">GW</span>
                            </p>
                            <p className="text-xs text-gray-500 mt-1">
                                Base: <span className="font-mono text-gray-400">{exp.base_value.toFixed(4)}</span>
                                {" · "}Δ:{" "}
                                <span className="font-mono text-gray-400">
                                    {(exp.predicted_demand - exp.base_value).toFixed(4)}
                                </span>
                            </p>
                        </div>
                    );
                })}
            </div>

            {/* ── Feature importance ────────────────────────────────────────────── */}
            {visibleAndAvailable.length > 0 && (
                <div className="bg-gray-900/30 border border-gray-800 rounded-xl p-5">
                    <h4 className="text-sm font-semibold text-white mb-1">Feature Importance</h4>
                    <p className="text-xs text-gray-500 mb-4">
                        Mean |SHAP| per feature across all {windowSize} context timesteps — which variable mattered
                        most?
                    </p>
                    <FeatureImportanceChart
                        data={artifactMap}
                        selectedDate={effectiveDate}
                        visibleVariants={visibleAndAvailable}
                    />
                </div>
            )}

            {/* ── Temporal attribution ─────────────────────────────────────────── */}
            {visibleAndAvailable.length > 0 && (
                <div className="bg-gray-900/30 border border-gray-800 rounded-xl p-5">
                    <h4 className="text-sm font-semibold text-white mb-1">Temporal Attribution</h4>
                    <p className="text-xs text-gray-500 mb-4">
                        Mean |SHAP| per past day — which of the last {windowSize} days drove this prediction?
                        D-1 = yesterday, D-{windowSize} = {windowSize} days ago.
                    </p>
                    <TimestepChart
                        data={artifactMap}
                        selectedDate={effectiveDate}
                        visibleVariants={visibleAndAvailable}
                        windowSize={windowSize}
                    />
                </div>
            )}

            {/* ── SHAP heatmaps ────────────────────────────────────────────────── */}
            {visibleAndAvailable.length > 0 && (
                <div className="bg-gray-900/30 border border-gray-800 rounded-xl p-5">
                    <h4 className="text-sm font-semibold text-white mb-1">SHAP Heatmap</h4>
                    <p className="text-xs text-gray-500 mb-6">
                        Raw SHAP matrix ({windowSize} days × {featureNames.length} features).{" "}
                        <span className="text-red-400">Red</span> = pushed prediction up,{" "}
                        <span className="text-blue-400">Blue</span> = pushed prediction down.
                        Intensity = magnitude.
                    </p>
                    <div className="flex gap-8 overflow-x-auto pb-2">
                        {visibleAndAvailable.map((v) => {
                            const exp = explanationFor(v);
                            if (!exp) return null;
                            return (
                                <ShapHeatmap
                                    key={v}
                                    explanation={exp}
                                    featureNames={featureNames}
                                    variantLabel={VARIANT_CONFIG[v].label}
                                    variantColor={VARIANT_CONFIG[v].color}
                                />
                            );
                        })}
                    </div>
                </div>
            )}

            {/* ── AI narrative (LLM) — one card per visible+available variant ───── */}
            {visibleAndAvailable.length > 0 && (
                <div className="space-y-4">
                    <h4 className="text-sm font-semibold text-white">AI Narrative — {effectiveDate}</h4>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {visibleAndAvailable.map((v) => (
                            <NarrativeSummary
                                key={v}
                                variant={v}
                                entry={get(v, effectiveDate)}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}