import {useReducer, useCallback, useEffect} from "react";
import {predictionsApi} from "@/lib/api";
import type {ForecastPoint, HistoricalPoint} from "@/types";

// ── State ─────────────────────────────────────────────────────────────────────
interface ForecastState {
    forecast: ForecastPoint[];
    historical: HistoricalPoint[];
    loading: boolean;
    error: string | null;
}

const INITIAL_STATE: ForecastState = {
    forecast: [],
    historical: [],
    loading: true,
    error: null,
};

// ── Actions ───────────────────────────────────────────────────────────────────
type ForecastAction =
    | { type: "LOAD_START" }
    | { type: "LOAD_DONE"; forecast: ForecastPoint[]; historical: HistoricalPoint[] }
    | { type: "LOAD_ERROR"; error: string };

function forecastReducer(state: ForecastState, action: ForecastAction): ForecastState {
    switch (action.type) {
        case "LOAD_START":
            return {...state, loading: true, error: null};
        case "LOAD_DONE":
            return {...state, loading: false, forecast: action.forecast, historical: action.historical};
        case "LOAD_ERROR":
            return {...state, loading: false, error: action.error};
    }
}

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useForecast() {
    const [state, dispatch] = useReducer(forecastReducer, INITIAL_STATE);

    // dispatch is stable — [] deps are correct.
    const load = useCallback(async (days = 60) => {
        dispatch({type: "LOAD_START"});
        try {
            const [f, h] = await Promise.all([
                predictionsApi.getForecast(),
                predictionsApi.getHistorical(days),
            ]);
            dispatch({type: "LOAD_DONE", forecast: f.data, historical: h.data});
        } catch {
            dispatch({type: "LOAD_ERROR", error: "Failed to load data. Please try again."});
        }
    }, []);

    // refresh invalidates the server cache then reloads — awaits load() intentionally
    const refresh = useCallback(async () => {
        await predictionsApi.refreshData();
        await load();
    }, [load]);

    // void: effect can't return a Promise; dispatch inside load is not flagged as setState
    useEffect(() => {
        void load();
    }, [load]);

    return {...state, refresh};
}