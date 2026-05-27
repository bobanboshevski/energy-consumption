"use client";

import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import type {ShapArtifact, ShapVariant} from "@/types/explainability";
import {VARIANT_CONFIG, FEATURE_LABELS} from "./constants";

interface Props {
    data: Partial<Record<ShapVariant, ShapArtifact>>;
    selectedDate: string;
    visibleVariants: ShapVariant[];
}

export function FeatureImportanceChart({data, selectedDate, visibleVariants}: Props) {
    // Build one data point per feature with a bar per variant
    const features = Object.keys(FEATURE_LABELS);

    const chartData = features.map((feat) => {
        const point: Record<string, any> = {feature: FEATURE_LABELS[feat] ?? feat};
        for (const variant of visibleVariants) {
            const artifact = data[variant];
            if (!artifact) continue;
            const explanation = artifact.explanations.find((e) => e.date === selectedDate);
            if (!explanation) continue;
            point[variant] = (explanation.feature_importance as any)[feat] ?? 0;
        }
        return point;
    });

    const CustomTooltip = ({active, payload, label}: any) => {
        if (!active || !payload?.length) return null;
        return (
            <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
                <p className="text-xs text-gray-400 mb-2 font-medium">{label}</p>
                {payload.map((p: any) => (
                    <div key={p.dataKey} className="flex items-center gap-2 text-sm">
                        <span className="w-2 h-2 rounded-full" style={{background: p.fill}}/>
                        <span className="text-gray-300">{VARIANT_CONFIG[p.dataKey as ShapVariant]?.label}:</span>
                        <span className="font-mono text-white">{Number(p.value).toFixed(5)}</span>
                    </div>
                ))}
            </div>
        );
    };

    return (
        <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData} margin={{top: 5, right: 20, left: 0, bottom: 5}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false}/>
                <XAxis dataKey="feature" tick={{fontSize: 11, fill: "#9ca3af"}}/>
                <YAxis tick={{fontSize: 10, fill: "#6b7280"}} tickFormatter={(v) => v.toFixed(3)}/>
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