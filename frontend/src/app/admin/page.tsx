"use client";

import {useState} from "react";
import {useAdmin} from "@/hooks/useAdmin";
import {MetricsGrid} from "@/components/admin/MetricsGrid";
import {PerformanceChart} from "@/components/admin/PerformanceChart";
import {ModelRegistry} from "@/components/admin/ModelRegistry";
import {ExperimentsTable} from "@/components/admin/ExperimentsTable";
import {DriftStatus} from "@/components/admin/DriftStatus";
import {Card, CardHeader} from "@/components/ui/Card";
import {LoadingSpinner} from "@/components/ui/LoadingSpinner";

type Tab = "monitoring" | "models" | "experiments";

const tabs: { id: Tab; label: string; icon: string }[] = [
    {id: "monitoring", label: "Monitoring", icon: "📊"},
    {id: "models", label: "Models", icon: "🤖"},
    {id: "experiments", label: "Experiments", icon: "🧪"},
];

export default function AdminPage() {
    const [tab, setTab] = useState<Tab>("monitoring");
    const {
        metrics, performance, drift, models, runs,
        loading, windowDays, setWindowDays,
        reload,
    } = useAdmin();

    if (loading) return <LoadingSpinner text="Loading admin dashboard..."/>;

    return (
        <div className="max-w-7xl mx-auto px-6 py-8">
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

            <div className="flex gap-2 mb-8 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit">
                {tabs.map((t) => (
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

            {tab === "monitoring" && (
                <div className="space-y-6">
                    {/*todo: here the backend is a bit confusing - it only calculates the metric on the last 30 days.*/}
                    {metrics && <MetricsGrid metrics={metrics}/>}

                    <Card>
                        <CardHeader
                            title="Model Performance Over Time"
                            subtitle="Actual vs predicted energy demand"
                        />
                        <div className="px-6 pb-6">
                            <PerformanceChart
                                data={performance}
                                windowDays={windowDays}
                                onWindowChange={setWindowDays}
                            />
                        </div>
                    </Card>

                    <Card>
                        <CardHeader title="Data Drift Report" subtitle="Latest Evidently drift analysis"/>
                        <div className="px-6 pb-6">
                            {drift
                                ? <DriftStatus drift={drift}/>
                                : <p className="text-gray-500 text-sm py-4">Drift data unavailable.</p>
                            }
                        </div>
                    </Card>
                </div>
            )}

            {tab === "models" && (
                <Card>
                    <CardHeader
                        title="Model Registry"
                        subtitle="Manage model versions and promotions"
                    />
                    <div className="px-6 pb-6">
                        <ModelRegistry models={models}/>
                    </div>
                </Card>
            )}

            {tab === "experiments" && (
                <Card>
                    <CardHeader
                        title="Experiment Runs"
                        subtitle="Training history from MLflow"
                    />
                    <div className="px-6 pb-6">
                        <ExperimentsTable runs={runs}/>
                    </div>
                </Card>
            )}
        </div>
    );
}