"use client";
import {useReducer, useCallback, useRef} from "react";
import {explainabilityApi, extractApiError} from "@/lib/explainabilityApi";
import type {ShapNarrative, ShapVariant} from "@/types/explainability";

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

export function useShapNarratives(version?: string) {
    const [state, dispatch] = useReducer(reducer, {entries: {}});

    // Keys already started (loading or done). A ref so fetchFor stays stable
    // (no state.entries dependency → no effect churn, no stale closures).
    const startedRef = useRef<Set<string>>(new Set());
    // The date of the most recent request — lets an in-progress sequential run
    // stop early when the admin switches to a different date.
    const activeDateRef = useRef<string | null>(null);

    const fetchFor = useCallback(
        (date: string, variants: ShapVariant[]) => {
            activeDateRef.current = date;

            const run = async () => {
                for (const variant of variants) {
                    if (activeDateRef.current !== date) return; // date changed — abandon this run
                    const key = keyOf(variant, date);
                    if (startedRef.current.has(key)) continue; // already loading or done

                    startedRef.current.add(key);
                    dispatch({type: "REQUEST", key});
                    try {
                        const r = await explainabilityApi.getNarrative(variant, date, version);
                        dispatch({
                            type: "SUCCESS",
                            key,
                            data: r.data.narrative,
                            predictedDemand: r.data.predicted_demand,
                        });
                    } catch (e) {
                        startedRef.current.delete(key); // failed → allow a future retry
                        dispatch({type: "FAILURE", key, error: extractApiError(e)});
                    }
                }
            };

            void run();
        },
        [version],
    );

    const get = useCallback(
        (variant: ShapVariant, date: string): NarrativeEntry | undefined => state.entries[keyOf(variant, date)],
        [state.entries],
    );

    return {get, fetchFor};
}