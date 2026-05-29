"use client";
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type {ShapArtifact, ShapVariant} from "@/types/explainability";
import {VARIANT_CONFIG} from "./constants";

// One chart point per timestep (D-30 … D-1).
// Variant keys are optional because not all variants may be visible.
interface TimestepChartPoint {
    label: string;
    keras?: number;
    onnx?: number;
    onnx_quantized?: number;
}

// Recharts tooltip content props — see PerformanceChart for full explanation.
// stroke is included here because Line components surface it on payload entries.
interface TooltipEntry {
    name?: string;
    value?: number | string;
    color?: string;
    stroke?: string;
    dataKey?: string | number;
}

interface ChartTooltipProps {
    active?: boolean;
    payload?: TooltipEntry[];
    label?: string | number;
}

// Defined at module level — NOT inside the component.
// Defining a component inside a render function creates a new component identity
// on every render, causing React to unmount and remount it each time.
const CustomTooltip = ({active, payload, label}: ChartTooltipProps) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
            <p className="text-xs text-gray-400 mb-2">{String(label)} (days before prediction)</p>
            {payload.map((entry) => {
                // dataKey is the ShapVariant string used as the Line's dataKey prop
                const variant = entry.dataKey as ShapVariant;
                const config = VARIANT_CONFIG[variant];
                return (
                    <div key={String(entry.dataKey)} className="flex items-center gap-2 text-sm">
                        <span
                            className="w-2 h-2 rounded-full"
                            style={{background: entry.stroke ?? entry.color}}
                        />
                        <span className="text-gray-300">{config?.label}:</span>
                        <span className="font-mono text-white">
                            {Number(entry.value).toFixed(5)}
                        </span>
                    </div>
                );
            })}
        </div>
    );
};

interface Props {
    data: Partial<Record<ShapVariant, ShapArtifact>>;
    selectedDate: string;
    visibleVariants: ShapVariant[];
    windowSize: number;
}

export function TimestepChart({data, selectedDate, visibleVariants, windowSize}: Props) {
    const chartData: TimestepChartPoint[] = Array.from({length: windowSize}, (_, i) => {
        const point: TimestepChartPoint = {label: `D-${windowSize - i}`};

        for (const variant of visibleVariants) {
            const artifact = data[variant];
            if (!artifact) continue;
            const explanation = artifact.explanations.find((e) => e.date === selectedDate);
            if (!explanation) continue;
            // variant is ShapVariant which matches the optional keys on TimestepChartPoint
            point[variant] = explanation.timestep_importance[i] ?? 0;
        }

        return point;
    });

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
                        fontSize: 11,
                    }}
                />
                <YAxis
                    tick={{fontSize: 10, fill: "#6b7280"}}
                    tickFormatter={(v: number) => v.toFixed(3)}
                />
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