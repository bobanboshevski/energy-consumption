"use client";

import type {ShapExplanation} from "@/types/explainability";
import {FEATURE_LABELS} from "./constants";

interface Props {
    explanation: ShapExplanation;
    featureNames: string[];
    variantLabel: string;
    variantColor: string;
}

const CELL_H = 11;
const CELL_W = 54;
const LABEL_W = 40;
const HEADER_H = 32;
const PADDING = 8;

/** Maps a SHAP value to a CSS rgba color on a dark background. */
function shapToColor(value: number, maxAbs: number): string {
    if (maxAbs < 1e-7 || Math.abs(value) < 1e-7) return "#111827";
    const intensity = Math.min(Math.abs(value) / maxAbs, 1);
    return value > 0
        ? `rgba(239, 68, 68, ${0.15 + intensity * 0.85})`   // red for positive
        : `rgba(59, 130, 246, ${0.15 + intensity * 0.85})`;  // blue for negative
}

function splitLabel(label: string): [string, string] {
    if (label.length <= 10) return [label, ""];
    const mid = Math.floor(label.length / 2);
    // Look for a space within ±4 chars of the midpoint
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < label.length; i++) {
        if (label[i] === " " || label[i] === "_" || label[i] === "-") {
            const dist = Math.abs(i - mid);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }
    }
    if (bestIdx !== -1) {
        return [label.slice(0, bestIdx), label.slice(bestIdx + 1)];
    }
    // No natural break — split at midpoint
    return [label.slice(0, mid), label.slice(mid)];
}

export function ShapHeatmap({explanation, featureNames, variantLabel, variantColor}: Props) {
    const matrix = explanation.shap_matrix;
    const windowSize = matrix.length;

    // Find global max absolute value for colour normalisation
    const maxAbs = Math.max(...matrix.flatMap((row) => row.map(Math.abs)));

    const svgWidth = LABEL_W + featureNames.length * CELL_W + PADDING;
    const svgHeight = HEADER_H + windowSize * CELL_H + PADDING;

    return (
        <div className="flex-1 min-w-0">
            {/* Variant label */}
            <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 rounded-full" style={{backgroundColor: variantColor}}/>
                <span className="text-xs font-medium" style={{color: variantColor}}>{variantLabel}</span>
            </div>

            <svg
                width={svgWidth}
                height={svgHeight}
                className="mb-2"
                style={{overflow: "visible", display: "block"}}
            >
                {/* Feature column headers */}

                {/*{featureNames.map((feat, fi) => (*/}
                {/*    */}
                {/*    <text*/}
                {/*        key={feat}*/}
                {/*        x={LABEL_W + fi * CELL_W + CELL_W / 2}*/}
                {/*        y={HEADER_H - 8}*/}
                {/*        textAnchor="middle"*/}
                {/*        fontSize={9}*/}
                {/*        fill="#9ca3af"*/}
                {/*    >*/}
                {/*        {FEATURE_LABELS[feat] ?? feat}*/}
                {/*    </text>*/}
                {/*))}*/}
                {featureNames.map((feat, fi) => {
                    const rawLabel = FEATURE_LABELS[feat] ?? feat;
                    const [line1, line2] = splitLabel(rawLabel);
                    const cx = LABEL_W + fi * CELL_W + CELL_W / 2;
                    return (
                        <text
                            key={feat}
                            x={cx}
                            textAnchor="middle"
                            fontSize={9}
                            fill="#9ca3af"
                        >
                            <tspan x={cx} y={HEADER_H - 20}>{line1}</tspan>
                            {line2 && <tspan x={cx} dy="11">{line2}</tspan>}
                        </text>
                    );
                })}

                {/* Cells */}
                {matrix.map((row, ti) => {
                    const y = HEADER_H + ti * CELL_H;
                    const isLast = ti === windowSize - 1;
                    return (
                        <g key={ti}>
                            {/* Timestep label — show every 5 steps + last */}
                            {(ti % 5 === 0 || isLast) && (
                                <text
                                    x={LABEL_W - 4}
                                    y={y + CELL_H / 2 + 3}
                                    textAnchor="end"
                                    fontSize={8}
                                    fill={isLast ? "#d1d5db" : "#6b7280"}
                                    fontWeight={isLast ? "600" : "400"}
                                >
                                    {isLast ? "D-1" : `D-${windowSize - ti}`}
                                </text>
                            )}

                            {row.map((val, fi) => (
                                <rect
                                    key={fi}
                                    x={LABEL_W + fi * CELL_W + 1}
                                    y={y + 1}
                                    width={CELL_W - 2}
                                    height={CELL_H - 2}
                                    rx={2}
                                    fill={shapToColor(val, maxAbs)}
                                >
                                    {/* SVG native tooltip — shown on hover by the browser */}
                                    <title>{`${featureNames[fi]}: ${val.toFixed(5)}`}</title>
                                </rect>

                            ))}
                        </g>
                    );
                })}

                {/* Colour legend */}
                {(() => {
                    const legendY = HEADER_H + windowSize * CELL_H + 4;
                    const legendW = featureNames.length * CELL_W;
                    const steps = 20;
                    const stepW = legendW / steps;
                    return (
                        <g>
                            {Array.from({length: steps}, (_, i) => {
                                const normalised = i / (steps - 1); // 0 → 1
                                const value = (normalised * 2 - 1) * maxAbs; // -maxAbs → +maxAbs
                                return (
                                    <rect
                                        key={i}
                                        x={LABEL_W + i * stepW}
                                        y={legendY}
                                        width={stepW}
                                        height={5}
                                        fill={shapToColor(value, maxAbs)}
                                    />
                                );
                            })}
                            <text x={LABEL_W} y={legendY + 14} fontSize={8} fill="#6b7280" textAnchor="start">
                                neg
                            </text>
                            <text x={LABEL_W + legendW} y={legendY + 14} fontSize={8} fill="#6b7280" textAnchor="end">
                                pos
                            </text>
                        </g>
                    );
                })()}
            </svg>
        </div>
    );
}