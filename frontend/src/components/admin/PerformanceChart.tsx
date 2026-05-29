"use client";
import {
    ComposedChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer, Area, Bar
} from "recharts";
import type {PerformancePoint} from "@/types";

function formatDate(d: string) {
    return new Date(d).toLocaleDateString("en-GB", {day: "numeric", month: "short"});
}

interface TooltipEntry {
    name?: string;
    value?: number | string;
    color?: string;
    dataKey?: string | number;
}

interface ChartTooltipProps {
    active?: boolean;
    payload?: TooltipEntry[];
    label?: string | number;
}

const CustomTooltip = ({active, payload, label}: ChartTooltipProps) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
            <p className="text-xs text-gray-400 mb-2">{formatDate(String(label))}</p>
            {payload.map((entry) => (
                <div key={String(entry.dataKey)} className="flex items-center gap-2 text-sm">
                    <span className="w-2 h-2 rounded-full" style={{background: entry.color}}/>
                    <span className="text-gray-300">{entry.name}:</span>
                    <span className="font-mono font-semibold text-white">
                        {Number(entry.value).toFixed(4)} GW
                    </span>
                </div>
            ))}
        </div>
    );
};

interface Props {
    data: PerformancePoint[];
    windowDays: number;
    onWindowChange: (days: number) => void;
}

export function PerformanceChart({data, windowDays, onWindowChange}: Props) {
    return (
        <div>
            <div className="flex gap-2 mb-4">
                {[7, 14, 30, 60, 90].map((d) => (
                    <button
                        key={d}
                        onClick={() => onWindowChange(d)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                            windowDays === d
                                ? "bg-blue-600 text-white"
                                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                        }`}
                    >
                        {d}d
                    </button>
                ))}
            </div>
            <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={data} margin={{top: 5, right: 20, left: 0, bottom: 20}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false}/>
                    <XAxis
                        dataKey="date"
                        tick={{fontSize: 11, fill: "#6b7280"}}
                        tickFormatter={formatDate}
                        interval={Math.floor(data.length / 6)}
                        label={{value: "Date", position: "insideBottom", offset: -10, fill: "#6b7280", fontSize: 12}}
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
                    <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={2} dot={false} name="Actual"/>
                    <Line type="monotone" dataKey="predicted" stroke="#f97316" strokeWidth={2} strokeDasharray="5 3"
                          dot={false} name="Predicted"/>
                    <Bar dataKey="error" fill="#374151" name="Error" opacity={0.5}/>
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}