"use client";

import {useReducer, useCallback} from "react";
import {explainabilityApi, extractApiError} from "@/lib/explainabilityApi";
import type {ShapArtifact, ExplainabilityErrors} from "@/types/explainability";

// ── State ──────────────────────────────────────────────────────────────────────

interface ExplainabilityState {
    keras: ShapArtifact | null;
    onnx: ShapArtifact | null;
    onnxQuantized: ShapArtifact | null;
    loading: boolean;
    errors: ExplainabilityErrors;
}

const INITIAL_STATE: ExplainabilityState = {
    keras: null,
    onnx: null,
    onnxQuantized: null,
    loading: false,
    errors: {keras: null, onnx: null, onnx_quantized: null},
};

// ── Actions ────────────────────────────────────────────────────────────────────

type Action =
    | { type: "LOADING" }
    | {
    type: "DONE";
    keras: ShapArtifact | null;
    onnx: ShapArtifact | null;
    onnxQuantized: ShapArtifact | null;
    errors: ExplainabilityErrors;
};

function reducer(state: ExplainabilityState, action: Action): ExplainabilityState {
    switch (action.type) {
        case "LOADING":
            return {...state, loading: true, errors: {keras: null, onnx: null, onnx_quantized: null}};
        case "DONE":
            return {
                keras: action.keras,
                onnx: action.onnx,
                onnxQuantized: action.onnxQuantized,
                loading: false,
                errors: action.errors,
            };
    }
}

// ── Hook ───────────────────────────────────────────────────────────────────────

/**
 * Loads SHAP explanations for all three model variants independently.
 * The three requests fire in parallel but are separate from all other admin requests.
 * Designed for lazy loading — call `load()` when the Explainability tab is first opened.
 */
export function useExplainability() {
    const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

    const load = useCallback(async (version?: string) => {
        dispatch({type: "LOADING"});

        const [kerasResult, onnxResult, onnxQResult] = await Promise.all([
            explainabilityApi.getKeras(version).catch((e) => ({error: e})),
            explainabilityApi.getOnnx(version).catch((e) => ({error: e})),
            explainabilityApi.getOnnxQuantized(version).catch((e) => ({error: e})),
        ]);

        const isError = (r: unknown): r is { error: unknown } =>
            typeof r === "object" && r !== null && "error" in r;

        dispatch({
            type: "DONE",
            keras: isError(kerasResult) ? null : (kerasResult as any).data,
            onnx: isError(onnxResult) ? null : (onnxResult as any).data,
            onnxQuantized: isError(onnxQResult) ? null : (onnxQResult as any).data,
            errors: {
                keras: isError(kerasResult) ? extractApiError(kerasResult.error) : null,
                onnx: isError(onnxResult) ? extractApiError(onnxResult.error) : null,
                onnx_quantized: isError(onnxQResult) ? extractApiError(onnxQResult.error) : null,
            },
        });
    }, []);

    return {...state, load};
}