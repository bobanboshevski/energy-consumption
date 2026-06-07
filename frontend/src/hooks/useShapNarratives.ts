"use client";
import {useReducer, useCallback} from "react";
import {explainabilityApi, extractApiError} from "@/lib/explainabilityApi";
import type {ShapNarrative, ShapVariant} from "@/types/explainability";

// One entry per (variant, date) key
type NarrativeEntry =
    | { status: "loading" }
    | { status: "done"; data: ShapNarrative; predictedDemand: number }
    | { status: "error"; error: string };

interface State {
    entries: Record<string, NarrativeEntry>;
}

type Action =
    | { type: "REQUEST"; key: string }
    | { type: "SUCCESS"; key: string; data: ShapNarrative; predictedDemand: number }
    | { type: "FAILURE"; key: string; error: string };

function reducer(state: State, action: Action): State {
    switch (action.type) {
        case "REQUEST":
            return {entries: {...state.entries, [action.key]: {status: "loading"}}};
        case "SUCCESS":
            return {
                entries: {
                    ...state.entries,
                    [action.key]: {status: "done", data: action.data, predictedDemand: action.predictedDemand},
                },
            };
        case "FAILURE":
            return {entries: {...state.entries, [action.key]: {status: "error", error: action.error}}};
    }
}

const keyOf = (variant: ShapVariant, date: string) => `${variant}:${date}`;

/**
 * On-demand LLM narratives, keyed by `${variant}:${date}`.
 * Kept separate from useExplainability — narratives are generated lazily,
 * only for the variants currently visible, and only when a date is selected.
 */
export function useShapNarratives(version?: string) {
    const [state, dispatch] = useReducer(reducer, {entries: {}});

    // Idempotent: skips keys already loading or done, so the effect can re-run freely.
    const fetchFor = useCallback(
        (date: string, variants: ShapVariant[]) => {
            for (const variant of variants) {
                const key = keyOf(variant, date);
                const existing = state.entries[key];
                if (existing && (existing.status === "loading" || existing.status === "done")) continue;

                dispatch({type: "REQUEST", key});
                explainabilityApi
                    .getNarrative(variant, date, version)
                    .then((r) =>
                        dispatch({
                            type: "SUCCESS",
                            key,
                            data: r.data.narrative,
                            predictedDemand: r.data.predicted_demand,
                        }),
                    )
                    .catch((e) => dispatch({type: "FAILURE", key, error: extractApiError(e)}));
            }
        },
        [state.entries, version],
    );

    const get = useCallback(
        (variant: ShapVariant, date: string): NarrativeEntry | undefined => state.entries[keyOf(variant, date)],
        [state.entries],
    );

    return {get, fetchFor};
}