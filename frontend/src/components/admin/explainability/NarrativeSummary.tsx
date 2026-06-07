"use client";
import type {ShapNarrative, ShapVariant} from "@/types/explainability";
import {VARIANT_CONFIG} from "./constants";

type Entry =
    | { status: "loading" }
    | { status: "done"; data: ShapNarrative; predictedDemand: number }
    | { status: "error"; error: string }
    | undefined;

export function NarrativeSummary({variant, entry}: { variant: ShapVariant; entry: Entry }) {
    const config = VARIANT_CONFIG[variant];

    return (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
                <span className="w-2.5 h-2.5 rounded-full" style={{background: config.color}}/>
                <span className="text-sm font-semibold text-white">{config.label}</span>
                <span className="text-xs text-gray-500 ml-auto">AI summary</span>
            </div>

            {(!entry || entry.status === "loading") && (
                <div className="flex items-center gap-3 py-6 text-gray-500 text-sm">
                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"/>
                    Generating narrative…
                </div>
            )}

            {entry?.status === "error" && (
                <p className="text-sm text-amber-400 py-4">
                    Narrative unavailable: {entry.error}
                </p>
            )}

            {entry?.status === "done" && (
                <div className="space-y-4">
                    <p className="text-sm text-white font-medium leading-relaxed">{entry.data.headline}</p>

                    <div className="grid grid-cols-3 gap-3 text-center">
                        <div className="bg-gray-950 rounded-lg py-2">
                            <p className="text-xs text-gray-500">Predicted</p>
                            <p className="text-sm font-mono font-semibold text-white">
                                {entry.predictedDemand.toFixed(4)} GW
                            </p>
                        </div>
                        <div className="bg-gray-950 rounded-lg py-2">
                            <p className="text-xs text-gray-500">Top feature</p>
                            <p className="text-sm font-semibold text-white">{entry.data.top_feature}</p>
                            <p className="text-xs text-gray-400">{entry.data.top_feature_share_pct?.toFixed(1)}%</p>
                        </div>
                        <div className="bg-gray-950 rounded-lg py-2">
                            <p className="text-xs text-gray-500">Key day</p>
                            <p className="text-sm font-semibold text-white">{entry.data.most_influential_day}</p>
                        </div>
                    </div>

                    {entry.data.key_findings?.length > 0 && (
                        <ul className="space-y-1.5">
                            {entry.data.key_findings.map((finding, i) => (
                                <li key={i} className="flex gap-2 text-sm text-gray-300">
                                    <span style={{color: config.color}}>•</span>
                                    <span>{finding}</span>
                                </li>
                            ))}
                        </ul>
                    )}

                    <p className="text-sm text-gray-400 leading-relaxed border-t border-gray-800 pt-3">
                        {entry.data.summary}
                    </p>
                </div>
            )}
        </div>
    );
}