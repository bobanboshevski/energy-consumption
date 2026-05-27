"use client";

import {useState} from "react";
import {useAdmin, type ModelView} from "@/hooks/useAdmin";
import {MetricsGrid} from "@/components/admin/MetricsGrid";
import {PerformanceChart} from "@/components/admin/PerformanceChart";
import {ModelRegistry} from "@/components/admin/ModelRegistry";
import {ModelSelector} from "@/components/admin/ModelSelector";
import {ExperimentsTable} from "@/components/admin/ExperimentsTable";
import {DriftStatus} from "@/components/admin/DriftStatus";
import {Card, CardHeader} from "@/components/ui/Card";
import {LoadingSpinner} from "@/components/ui/LoadingSpinner";
import {GxStatus} from "@/components/admin/GxStatus";
import {ComparisonChart} from "@/components/admin/ComparisonChart";
import {ExplainabilityPanel} from "@/components/admin/explainability/ExplainabilityPanel";

type Tab = "monitoring" | "models" | "experiments" | "explainability";

const TABS: { id: Tab; label: string; icon: string }[] = [
    {id: "monitoring", label: "Monitoring", icon: "📊"},
    {id: "models", label: "Models", icon: "🤖"},
    {id: "experiments", label: "Experiments", icon: "🧪"},
    {id: "explainability", label: "Explainability", icon: "🔍"},
];

export default function AdminPage() {
    const [tab, setTab] = useState<Tab>("monitoring");
    const [modelView, setModelView] = useState<ModelView>("multivariate");

    const {
        metrics, performance,
        univariateMetrics, univariatePerformance,
        drift, validationReport,
        models, runs, loadingModels,
        isInitialLoading, loadingMonitoring,
        windowDays, setWindowDays,
        reload, reloadModels,
        comparisonMultivariate, comparisonUnivariate, loadingComparison,
    } = useAdmin();

    if (isInitialLoading) return <LoadingSpinner text="Loading admin dashboard..."/>;

    const activeMetrics = modelView === "multivariate" ? metrics : univariateMetrics;
    const activePerformance = modelView === "multivariate" ? performance : univariatePerformance;
    const activeRuns = runs[modelView];

    return (
        <div className="max-w-7xl mx-auto px-6 py-8">
            {/* Header */}
            <div className="flex items-start justify-between mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
                    <p className="text-gray-400 mt-1">Model monitoring, registry and experiment tracking</p>
                </div>
                <button
                    onClick={reload}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl text-sm text-gray-300 transition-colors"
                >
                    ↻ Reload
                </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mb-6 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
                {TABS.map((t) => (
                    <button
                        key={t.id}
                        onClick={() => setTab(t.id)}
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                            tab === t.id
                                ? "bg-blue-600 text-white shadow-lg"
                                : "text-gray-400 hover:text-white hover:bg-gray-800"
                        }`}
                    >
                        <span>{t.icon}</span>
                        {t.label}
                    </button>
                ))}
            </div>

            {/* ── Monitoring tab ──────────────────────────────────────────────────── */}
            {tab === "monitoring" && (
                <div className="space-y-6">
                    <ModelSelector selected={modelView} onChange={setModelView}/>

                    {loadingMonitoring ? (
                        <LoadingSpinner text="Loading monitoring data..."/>
                    ) : (
                        <>
                            {activeMetrics && <MetricsGrid metrics={activeMetrics}/>}

                            <Card>
                                <CardHeader
                                    title="Model Performance Over Time"
                                    subtitle={`Actual vs predicted — ${modelView} model`}
                                />
                                <div className="px-6 pb-6">
                                    <PerformanceChart
                                        data={activePerformance}
                                        windowDays={windowDays}
                                        onWindowChange={setWindowDays}
                                    />
                                </div>
                            </Card>

                            {/* ── Backend comparison ───────────────────────────────────── */}
                            <Card>
                                <CardHeader
                                    title="Backend Comparison — ONNX vs Keras"
                                    subtitle={`Side-by-side accuracy comparison — ${modelView} model`}
                                />
                                <div className="px-6 pb-6">
                                    {loadingComparison ? (
                                        <div className="flex items-center gap-3 py-8 text-gray-500 text-sm">
                                            <div
                                                className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"/>
                                            Loading comparison data...
                                        </div>
                                    ) : (() => {
                                        const comparison = modelView === "multivariate"
                                            ? comparisonMultivariate
                                            : comparisonUnivariate;
                                        return comparison
                                            ? <ComparisonChart comparison={comparison}/>
                                            :
                                            <p className="text-gray-500 text-sm py-4">Comparison data unavailable.</p>;
                                    })()}
                                </div>
                            </Card>

                            {/* Drift only shown for multivariate (shared report) */}
                            {modelView === "multivariate" && (
                                <>
                                    <Card>
                                        <CardHeader title="Data Drift Report"
                                                    subtitle="Latest Evidently drift analysis"/>
                                        <div className="px-6 pb-6">
                                            {drift
                                                ? <DriftStatus drift={drift}/>
                                                : <p className="text-gray-500 text-sm py-4">Drift data unavailable.</p>
                                            }
                                        </div>
                                    </Card>

                                    <Card>
                                        <CardHeader
                                            title="Data Validation Report"
                                            subtitle="Latest Great Expectations checkpoint results"
                                        />
                                        <div className="px-6 pb-6">
                                            {validationReport
                                                ? <GxStatus report={validationReport}/>
                                                : <p className="text-gray-500 text-sm py-4">Validation report
                                                    unavailable.</p>
                                            }
                                        </div>
                                    </Card>
                                </>
                            )}
                        </>
                    )}
                </div>
            )}

            {/* ── Models tab ──────────────────────────────────────────────────────── */}
            {/*{tab === "models" && (*/}
            {/*    <Card>*/}
            {/*        <CardHeader*/}
            {/*            title="Model Registry"*/}
            {/*            subtitle="Manage versions and activate models for serving"*/}
            {/*        />*/}
            {/*        <div className="px-6 pb-6">*/}
            {/*            <ModelRegistry models={models} onReload={reloadModels}/>*/}
            {/*        </div>*/}
            {/*    </Card>*/}
            {/*)}*/}

            {tab === "models" && (
                <Card>
                    <CardHeader
                        title="Model Registry"
                        subtitle="Manage versions and activate models for serving"
                    />
                    <div className="px-6 pb-6">
                        <ModelRegistry
                            models={models}
                            loadingModels={loadingModels}
                            onReload={reloadModels}
                        />
                    </div>
                </Card>
            )}

            {/* ── Experiments tab ─────────────────────────────────────────────────── */}
            {tab === "experiments" && (
                <div className="space-y-6">
                    <ModelSelector selected={modelView} onChange={setModelView}/>
                    <Card>
                        <CardHeader
                            title="Experiment Runs"
                            subtitle={`Training history — ${modelView} model`}
                        />
                        <div className="px-6 pb-6">
                            <ExperimentsTable runs={activeRuns}/>
                        </div>
                    </Card>
                </div>
            )}

            {tab === "explainability" && (
                <Card>
                    <CardHeader
                        title="Model Explainability"
                        subtitle="SHAP feature attribution for multivariate model forecast predictions"
                    />
                    <div className="px-6 pb-6">
                        <ExplainabilityPanel/>
                    </div>
                </Card>
            )}

        </div>
    );
}