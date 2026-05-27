"use client";

import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type {ShapArtifact, ShapVariant} from "@/types/explainability";
import {VARIANT_CONFIG} from "./constants";

interface Props {
    data: Partial<Record<ShapVariant, ShapArtifact>>;
    selectedDate: string;
    visibleVariants: ShapVariant[];
    windowSize: number;
}

export function TimestepChart({data, selectedDate, visibleVariants, windowSize}: Props) {
    // Build one point per timestep — D-30 (index 0) to D-1 (index 29)
    const chartData = Array.from({length: windowSize}, (_, i) => {
        const label = `D-${windowSize - i}`;
        const point: Record<string, any> = {label};
        for (const variant of visibleVariants) {
            const artifact = data[variant];
            if (!artifact) continue;
            const explanation = artifact.explanations.find((e) => e.date === selectedDate);
            if (!explanation) continue;
            point[variant] = explanation.timestep_importance[i] ?? 0;
        }
        return point;
    });

    const CustomTooltip = ({active, payload, label}: any) => {
        if (!active || !payload?.length) return null;
        return (
            <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
                <p className="text-xs text-gray-400 mb-2">{label} (days before prediction)</p>
                {payload.map((p: any) => (
                    <div key={p.dataKey} className="flex items-center gap-2 text-sm">
                        <span className="w-2 h-2 rounded-full" style={{background: p.stroke}}/>
                        <span className="text-gray-300">{VARIANT_CONFIG[p.dataKey as ShapVariant]?.label}:</span>
                        <span className="font-mono text-white">{Number(p.value).toFixed(5)}</span>
                    </div>
                ))}
            </div>
        );
    };

    return (
        <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{top: 5, right: 20, left: 0, bottom: 5}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false}/>
                <XAxis
                    dataKey="label"
                    tick={{fontSize: 9, fill: "#6b7280"}}
                    interval={4}
                    label={{
                        value: "Days before prediction →",
                        position: "insideBottom",
                        offset: -4,
                        fill: "#6b7280",
                        fontSize: 11
                    }}
                />
                <YAxis tick={{fontSize: 10, fill: "#6b7280"}} tickFormatter={(v) => v.toFixed(3)}/>
                <Tooltip content={<CustomTooltip/>}/>
                <Legend
                    wrapperStyle={{fontSize: "12px", color: "#9ca3af", paddingTop: "8px"}}
                    formatter={(v) => VARIANT_CONFIG[v as ShapVariant]?.label ?? v}
                />
                {visibleVariants.map((v) => (
                    <Line
                        key={v}
                        dataKey={v}
                        stroke={VARIANT_CONFIG[v].color}
                        strokeWidth={1.5}
                        dot={false}
                        name={v}
                    />
                ))}
            </LineChart>
        </ResponsiveContainer>
    );
}