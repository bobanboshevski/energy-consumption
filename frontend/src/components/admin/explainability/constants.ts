import type {ShapVariant} from "@/types/explainability";

export const VARIANT_CONFIG: Record<ShapVariant, {
    label: string;
    color: string;
    badgeBg: string;
    badgeBorder: string;
    badgeText: string;
}> = {
    keras: {
        label: "Keras",
        color: "#3b82f6",
        badgeBg: "bg-blue-950/40",
        badgeBorder: "border-blue-800",
        badgeText: "text-blue-400",
    },
    onnx: {
        label: "ONNX",
        color: "#a855f7",
        badgeBg: "bg-purple-950/40",
        badgeBorder: "border-purple-800",
        badgeText: "text-purple-400",
    },
    onnx_quantized: {
        label: "ONNX Quantized",
        color: "#f59e0b",
        badgeBg: "bg-amber-950/40",
        badgeBorder: "border-amber-800",
        badgeText: "text-amber-400",
    },
};

export const ALL_VARIANTS: ShapVariant[] = ["keras", "onnx", "onnx_quantized"];

/** Maps raw feature key to display label. */
export const FEATURE_LABELS: Record<string, string> = {
    energy_demand: "Energy Demand",
    temp_max: "Temp Max",
    temp_min: "Temp Min",
    daylight_duration: "Daylight",
};