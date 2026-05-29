"use client";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type {ShapArtifact, ShapVariant} from "@/types/explainability";
import type {FeatureImportance} from "@/types/explainability";
import {VARIANT_CONFIG, FEATURE_LABELS} from "./constants";

// One chart point per feature — one bar per visible variant.
interface FeatureImportanceChartPoint {
    feature: string;        // display label e.g. "Energy Demand"
    keras?: number;
    onnx?: number;
    onnx_quantized?: number;
}

// Recharts tooltip content props.
// fill is included because Bar components surface their fill colour on payload entries.
interface TooltipEntry {
    name?: string;
    value?: number | string;
    color?: string;
    fill?: string;
    dataKey?: string | number;
}

interface ChartTooltipProps {
    active?: boolean;
    payload?: TooltipEntry[];
    label?: string | number;
}

// Defined at module level — NOT inside the component.
// Defining a component inside a render function creates a new identity on every
// render, causing React to unmount and remount it (and lose tooltip state).
const CustomTooltip = ({active, payload, label}: ChartTooltipProps) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
            <p className="text-xs text-gray-400 mb-2 font-medium">{String(label)}</p>
            {payload.map((entry) => {
                // dataKey is the ShapVariant string used as the Bar's dataKey prop
                const variant = entry.dataKey as ShapVariant;
                const config = VARIANT_CONFIG[variant];
                return (
                    <div key={String(entry.dataKey)} className="flex items-center gap-2 text-sm">
                        <span
                            className="w-2 h-2 rounded-full"
                            style={{background: entry.fill ?? entry.color}}
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
}

export function FeatureImportanceChart({data, selectedDate, visibleVariants}: Props) {
    const chartData: FeatureImportanceChartPoint[] = Object.keys(FEATURE_LABELS).map((feat) => {
        // feat is a key of both FEATURE_LABELS and FeatureImportance — cast is safe
        const featKey = feat as keyof FeatureImportance;
        const point: FeatureImportanceChartPoint = {feature: FEATURE_LABELS[feat] ?? feat};

        for (const variant of visibleVariants) {
            const artifact = data[variant];
            if (!artifact) continue;
            const explanation = artifact.explanations.find((e) => e.date === selectedDate);
            if (!explanation) continue;
            // featKey is keyof FeatureImportance, so this access is fully typed
            point[variant] = explanation.feature_importance[featKey] ?? 0;
        }

        return point;
    });

    return (
        <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData} margin={{top: 5, right: 20, left: 0, bottom: 5}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false}/>
                <XAxis dataKey="feature" tick={{fontSize: 11, fill: "#9ca3af"}}/>
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
                    <Bar
                        key={v}
                        dataKey={v}
                        fill={VARIANT_CONFIG[v].color}
                        name={v}
                        radius={[3, 3, 0, 0]}
                        opacity={0.85}
                    />
                ))}
            </BarChart>
        </ResponsiveContainer>
    );
}