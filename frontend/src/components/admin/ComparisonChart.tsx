"use client";

import {
    ComposedChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type {BackendComparison} from "@/types";

function formatDate(d: string) {
    return new Date(d).toLocaleDateString("en-GB", {day: "numeric", month: "short"});
}

const CustomTooltip = ({active, payload, label}: any) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
            <p className="text-xs text-gray-400 mb-2">{formatDate(label)}</p>
            {payload.map((p: any) => (
                <div key={p.name} className="flex items-center gap-2 text-sm">
                    <span className="w-2 h-2 rounded-full" style={{background: p.color}}/>
                    <span className="text-gray-300">{p.name}:</span>
                    <span className="font-mono font-semibold text-white">{Number(p.value).toFixed(4)} GW</span>
                </div>
            ))}
        </div>
    );
};

function MetricPill({label, onnxVal, kerasVal}: {
    label: string;
    onnxVal?: number;
    kerasVal?: number;
}) {
    const diff = onnxVal !== undefined && kerasVal !== undefined
        ? ((onnxVal - kerasVal) / kerasVal * 100).toFixed(1)
        : null;

    const diffColor = diff === null
        ? "text-gray-500"
        : parseFloat(diff) > 0
            ? "text-red-400"
            : "text-emerald-400";

    return (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">{label}</p>
            <div className="space-y-2">
                {onnxVal !== undefined && (
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-purple-400 font-medium">ONNX</span>
                        <span className="font-mono text-sm text-white font-semibold">{onnxVal}</span>
                    </div>
                )}
                {kerasVal !== undefined && (
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-blue-400 font-medium">Keras</span>
                        <span className="font-mono text-sm text-white font-semibold">{kerasVal}</span>
                    </div>
                )}
                {diff !== null && (
                    <div className={`text-xs text-right ${diffColor} font-mono`}>
                        {parseFloat(diff) > 0 ? "+" : ""}{diff}% vs Keras
                    </div>
                )}
            </div>
        </div>
    );
}

interface Props {
    comparison: BackendComparison;
}

export function ComparisonChart({comparison}: Props) {
    const {onnx, keras} = comparison;

    // Merge performance points by date for the chart
    const dateMap: Record<string, any> = {};

    if (keras.available) {
        keras.performance.forEach((p) => {
            dateMap[p.date] = {date: p.date, actual: p.actual, keras: p.predicted};
        });
    }

    if (onnx.available) {
        onnx.performance.forEach((p) => {
            if (!dateMap[p.date]) dateMap[p.date] = {date: p.date};
            dateMap[p.date].onnx = p.predicted;
            if (!dateMap[p.date].actual) dateMap[p.date].actual = p.actual;
        });
    }

    const chartData = Object.values(dateMap).sort(
        (a: any, b: any) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const metricKeys = ["mae", "mse", "rmse", "mean_error", "max_error"] as const;

    if (!onnx.available && !keras.available) {
        return (
            <p className="text-gray-500 text-sm py-8 text-center">
                No comparison data available — models have not been loaded yet.
            </p>
        );
    }

    return (
        <div className="space-y-6">
            {/* Status badges */}
            <div className="flex gap-3 flex-wrap">
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium ${
                    onnx.available
                        ? "bg-purple-950/40 border-purple-800 text-purple-300"
                        : "bg-gray-800 border-gray-700 text-gray-500"
                }`}>
                    <span className={`w-2 h-2 rounded-full ${onnx.available ? "bg-purple-400" : "bg-gray-600"}`}/>
                    ONNX Runtime {onnx.variant ? `(${onnx.variant})` : ""} — {onnx.available ? "active" : "unavailable"}
                </div>
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium ${
                    keras.available
                        ? "bg-blue-950/40 border-blue-800 text-blue-300"
                        : "bg-gray-800 border-gray-700 text-gray-500"
                }`}>
                    <span className={`w-2 h-2 rounded-full ${keras.available ? "bg-blue-400" : "bg-gray-600"}`}/>
                    Keras / TensorFlow — {keras.available ? "loaded for comparison" : "unavailable"}
                </div>
            </div>

            {/* Metrics comparison grid */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {metricKeys.map((k) => (
                    <MetricPill
                        key={k}
                        label={k.replace("_", " ").toUpperCase()}
                        onnxVal={onnx.available ? onnx.metrics[k] : undefined}
                        kerasVal={keras.available ? keras.metrics[k] : undefined}
                    />
                ))}
            </div>

            {/* Overlay chart */}
            <div>
                <p className="text-xs text-gray-500 mb-3 uppercase tracking-wider">
                    Predicted values — same evaluation points, both backends
                </p>
                <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={chartData} margin={{top: 5, right: 20, left: 0, bottom: 20}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false}/>
                        <XAxis
                            dataKey="date"
                            tick={{fontSize: 11, fill: "#6b7280"}}
                            tickFormatter={formatDate}
                            interval={Math.floor(chartData.length / 5)}
                            label={{
                                value: "Date",
                                position: "insideBottom",
                                offset: -10,
                                fill: "#6b7280",
                                fontSize: 12
                            }}
                        />
                        <YAxis
                            tick={{fontSize: 11, fill: "#6b7280"}}
                            label={{
                                value: "GW",
                                angle: -90,
                                position: "insideLeft",
                                offset: 15,
                                fill: "#6b7280",
                                fontSize: 12
                            }}
                        />
                        <Tooltip content={<CustomTooltip/>}/>
                        <Legend wrapperStyle={{fontSize: "12px", color: "#9ca3af", paddingTop: "12px"}}/>
                        <Line
                            type="monotone"
                            dataKey="actual"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={false}
                            name="Actual"
                        />
                        {keras.available && (
                            <Line
                                type="monotone"
                                dataKey="keras"
                                stroke="#60a5fa"
                                strokeWidth={1.5}
                                strokeDasharray="6 3"
                                dot={false}
                                name="Keras"
                            />
                        )}
                        {onnx.available && (
                            <Line
                                type="monotone"
                                dataKey="onnx"
                                stroke="#a855f7"
                                strokeWidth={1.5}
                                strokeDasharray="3 3"
                                dot={false}
                                name={`ONNX (${onnx.variant ?? "base"})`}
                            />
                        )}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}