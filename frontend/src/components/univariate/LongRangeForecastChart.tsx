"use client";

import {useState, useEffect} from "react";
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend
} from "recharts";
import {useLongRangeForecast} from "@/hooks/useUnivariate";
import {InlineSpinner} from "@/components/ui/LoadingSpinner";
import type {DemandCategory, UnivariateRangePoint} from "@/types";

const PRESETS: { label: string; months: number }[] = [
    {label: "1 month", months: 1},
    {label: "3 months", months: 3},
    {label: "6 months", months: 6},
    {label: "12 months", months: 12},
];

function addMonths(date: Date, months: number): Date {
    const d = new Date(date);
    d.setMonth(d.getMonth() + months);
    return d;
}

function toDateStr(d: Date): string {
    return d.toLocaleDateString("en-CA");
}

function formatDate(d: string) {
    return new Date(d).toLocaleDateString("en-GB", {day: "numeric", month: "short"});
}

const categoryColor: Record<DemandCategory, string> = {
    low: "#22c55e",
    medium: "#f97316",
    high: "#ef4444",
};

interface TooltipEntry {
    name?: string;
    value?: number | string;
    color?: string;
    dataKey?: string | number;
    payload?: UnivariateRangePoint;   // ← the actual chart data row
}

interface ChartTooltipProps {
    active?: boolean;
    payload?: TooltipEntry[];
    label?: string | number;
}

const CustomTooltip = ({active, payload, label}: ChartTooltipProps) => {
    if (!active || !payload?.length) return null;
    const point = payload[0]?.payload;
    return (
        <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
            <p className="text-xs text-gray-400 mb-1">{formatDate(String(label))}</p>
            <p className="text-sm font-mono font-semibold text-white">
                {Number(payload[0]?.value).toFixed(4)} GW
            </p>
            {point?.demand_category && (
                <p className="text-xs mt-1 capitalize"
                   style={{color: categoryColor[point.demand_category as DemandCategory]}}>
                    {point.demand_category} demand
                </p>
            )}
            <p className="text-xs text-gray-600 mt-1">{point?.days_ahead}d ahead</p>
        </div>
    );
};

export function LongRangeForecastChart() {
    const [selectedMonths, setSelectedMonths] = useState(3);
    const {data, loading, error, load} = useLongRangeForecast();

    const loadRange = (months: number) => {
        const today = new Date();
        const start = new Date(today);
        start.setDate(today.getDate() + 1);
        const end = addMonths(today, months);
        load(toDateStr(start), toDateStr(end));
    };

    useEffect(() => {
        loadRange(selectedMonths);
    }, []);

    const handlePresetChange = (months: number) => {
        setSelectedMonths(months);
        loadRange(months);
    };

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <div className="flex gap-2">
                    {PRESETS.map((p) => (
                        <button
                            key={p.months}
                            onClick={() => handlePresetChange(p.months)}
                            disabled={loading}
                            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 ${
                                selectedMonths === p.months
                                    ? "bg-purple-600 text-white"
                                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                            }`}
                        >
                            {p.label}
                        </button>
                    ))}
                </div>
                {loading && (
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                        <InlineSpinner/> Computing...
                    </div>
                )}
            </div>

            {error && (
                <div className="p-3 bg-red-950/40 border border-red-800/50 rounded-xl">
                    <p className="text-sm text-red-400">{error}</p>
                </div>
            )}

            {!loading && data.length > 0 && (
                <>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={data} margin={{top: 5, right: 20, left: 0, bottom: 20}}>
                            <defs>
                                <linearGradient id="univariateGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.2}/>
                                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false}/>
                            <XAxis
                                dataKey="date"
                                tick={{fontSize: 10, fill: "#6b7280"}}
                                tickFormatter={formatDate}
                                interval={Math.floor(data.length / 6)}
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
                                domain={["auto", "auto"]}
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
                            <Area
                                type="monotone"
                                dataKey="predicted_demand"
                                stroke="#a855f7"
                                strokeWidth={2}
                                fill="url(#univariateGrad)"
                                dot={false}
                                name="Long-range Forecast"
                            />
                        </AreaChart>
                    </ResponsiveContainer>

                    <div className="grid grid-cols-3 gap-3">
                        {(["low", "medium", "high"] as const).map((cat) => {
                            const count = data.filter((d) => d.demand_category === cat).length;
                            const pct = Math.round((count / data.length) * 100);
                            return (
                                <div key={cat} className="bg-gray-800/50 rounded-xl p-3 text-center">
                                    <div className="w-2 h-2 rounded-full mx-auto mb-1.5"
                                         style={{background: categoryColor[cat]}}/>
                                    <p className="text-xs text-gray-500 capitalize">{cat} demand</p>
                                    <p className="font-mono text-sm font-semibold text-white mt-0.5">{pct}%</p>
                                    <p className="text-xs text-gray-600">{count} days</p>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}

            {loading && (
                <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
                    Loading {selectedMonths}-month forecast...
                </div>
            )}
        </div>
    );
}