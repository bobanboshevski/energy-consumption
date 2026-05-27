import {api} from "@/lib/api";
import axios from "axios";
import {ShapArtifact} from "@/types/explainability";

function params(version?: string) {
    return version ? {params: {version}} : {};
}

export const explainabilityApi = {
    getKeras: (version?: string) =>
        api.get<ShapArtifact>("/explainability/keras", params(version)),

    getOnnx: (version?: string) =>
        api.get<ShapArtifact>("/explainability/onnx", params(version)),

    getOnnxQuantized: (version?: string) =>
        api.get<ShapArtifact>("/explainability/onnx_quantized", params(version)),
};

/** Extracts a human-readable message from an axios error. */
export function extractApiError(error: unknown): string {
    if (axios.isAxiosError(error)) {
        const detail = error.response?.data?.detail;
        if (typeof detail === "string") return detail;
        if (detail?.message) return detail.message;
        if (error.response?.status === 503) return "MLflow is unavailable. Try again later.";
        if (error.response?.status === 404)
            return "Explanations not available for this model version. Re-train to generate them.";
    }
    return "Failed to load explanations.";
}