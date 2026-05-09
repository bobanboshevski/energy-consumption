// "use client";
//
// import {useState} from "react";
// import {useAdmin} from "@/hooks/useAdmin";
// import {MetricsGrid} from "@/components/admin/MetricsGrid";
// import {PerformanceChart} from "@/components/admin/PerformanceChart";
// import {ModelRegistry} from "@/components/admin/ModelRegistry";
// import {ExperimentsTable} from "@/components/admin/ExperimentsTable";
// import {DriftStatus} from "@/components/admin/DriftStatus";
// import {Card, CardHeader} from "@/components/ui/Card";
// import {LoadingSpinner} from "@/components/ui/LoadingSpinner";
//
// type Tab = "monitoring" | "models" | "experiments";
//
// const tabs: { id: Tab; label: string; icon: string }[] = [
//     {id: "monitoring", label: "Monitoring", icon: "📊"},
//     {id: "models", label: "Models", icon: "🤖"},
//     {id: "experiments", label: "Experiments", icon: "🧪"},
// ];
//
// export default function AdminPage() {
//     const [tab, setTab] = useState<Tab>("monitoring");
//     const {
//         metrics, performance, drift, models, runs,
//         loading, windowDays, setWindowDays,
//         reload,
//     } = useAdmin();
//
//     if (loading) return <LoadingSpinner text="Loading admin dashboard..."/>;
//
//     return (
//         <div className="max-w-7xl mx-auto px-6 py-8">
//             <div className="flex items-start justify-between mb-8">
//                 <div>
//                     <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
//                     <p className="text-gray-400 mt-1">Model monitoring, registry and experiment tracking</p>
//                 </div>
//                 <button
//                     onClick={reload}
//                     className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl text-sm text-gray-300 transition-colors"
//                 >
//                     ↻ Reload
//                 </button>
//             </div>
//
//             <div className="flex gap-2 mb-8 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
//                 {tabs.map((t) => (
//                     <button
//                         key={t.id}
//                         onClick={() => setTab(t.id)}
//                         className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
//                             tab === t.id
//                                 ? "bg-blue-600 text-white shadow-lg"
//                                 : "text-gray-400 hover:text-white hover:bg-gray-800"
//                         }`}
//                     >
//                         <span>{t.icon}</span>
//                         {t.label}
//                     </button>
//                 ))}
//             </div>
//
//             {tab === "monitoring" && (
//                 <div className="space-y-6">
//                     {/*todo: here the backend is a bit confusing - it only calculates the metric on the last 30 days.*/}
//                     {metrics && <MetricsGrid metrics={metrics}/>}
//
//                     <Card>
//                         <CardHeader
//                             title="Model Performance Over Time"
//                             subtitle="Actual vs predicted energy demand"
//                         />
//                         <div className="px-6 pb-6">
//                             <PerformanceChart
//                                 data={performance}
//                                 windowDays={windowDays}
//                                 onWindowChange={setWindowDays}
//                             />
//                         </div>
//                     </Card>
//
//                     <Card>
//                         <CardHeader title="Data Drift Report" subtitle="Latest Evidently drift analysis"/>
//                         <div className="px-6 pb-6">
//                             {drift
//                                 ? <DriftStatus drift={drift}/>
//                                 : <p className="text-gray-500 text-sm py-4">Drift data unavailable.</p>
//                             }
//                         </div>
//                     </Card>
//                 </div>
//             )}
//
//             {tab === "models" && (
//                 <Card>
//                     <CardHeader
//                         title="Model Registry"
//                         subtitle="Manage model versions and promotions"
//                     />
//                     <div className="px-6 pb-6">
//                         <ModelRegistry models={models}/>
//                     </div>
//                 </Card>
//             )}
//
//             {tab === "experiments" && (
//                 <Card>
//                     <CardHeader
//                         title="Experiment Runs"
//                         subtitle="Training history from MLflow"
//                     />
//                     <div className="px-6 pb-6">
//                         <ExperimentsTable runs={runs}/>
//                     </div>
//                 </Card>
//             )}
//         </div>
//     );
// }

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

type Tab = "monitoring" | "models" | "experiments";

const TABS: { id: Tab; label: string; icon: string }[] = [
    {id: "monitoring", label: "Monitoring", icon: "📊"},
    {id: "models", label: "Models", icon: "🤖"},
    {id: "experiments", label: "Experiments", icon: "🧪"},
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
        </div>
    );
}