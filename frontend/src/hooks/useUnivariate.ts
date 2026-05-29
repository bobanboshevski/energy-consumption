import axios from "axios";
import {useReducer, useCallback, useEffect} from "react";
import {univariateApi} from "@/lib/api";
import type {UnivariatePrediction, UnivariateRangePoint, UnivariateModelInfo} from "@/types";

// ── useUnivariateInfo ─────────────────────────────────────────────────────────

interface InfoState {
    info: UnivariateModelInfo | null;
    loading: boolean;
}

type InfoAction =
    | { type: "LOAD_START" }
    | { type: "LOAD_DONE"; info: UnivariateModelInfo };

function infoReducer(state: InfoState, action: InfoAction): InfoState {
    switch (action.type) {
        case "LOAD_START":
            return {...state, loading: true};
        case "LOAD_DONE":
            return {info: action.info, loading: false};
    }
}

export function useUnivariateInfo() {
    const [state, dispatch] = useReducer(infoReducer, {info: null, loading: true});

    const load = useCallback(async () => {
        dispatch({type: "LOAD_START"});
        try {
            const r = await univariateApi.getInfo();
            dispatch({type: "LOAD_DONE", info: r.data});
        } catch {
            // info is non-critical — silently fail, leave loading: false
            dispatch({type: "LOAD_DONE", info: null as unknown as UnivariateModelInfo});
        }
    }, []);

    useEffect(() => {
        void load();
    }, [load]);

    return {info: state.info, loading: state.loading};
}

// ── useDatePrediction ─────────────────────────────────────────────────────────

interface DatePredictionState {
    result: UnivariatePrediction | null;
    loading: boolean;
    error: string | null;
}

type DatePredictionAction =
    | { type: "PREDICT_START" }
    | { type: "PREDICT_DONE"; result: UnivariatePrediction }
    | { type: "PREDICT_ERROR"; error: string }
    | { type: "RESET" };

function datePredictionReducer(state: DatePredictionState, action: DatePredictionAction): DatePredictionState {
    switch (action.type) {
        case "PREDICT_START":
            return {result: null, loading: true, error: null};
        case "PREDICT_DONE":
            return {result: action.result, loading: false, error: null};
        case "PREDICT_ERROR":
            return {result: null, loading: false, error: action.error};
        case "RESET":
            return {result: null, loading: false, error: null};
    }
}

export function useDatePrediction() {
    const [state, dispatch] = useReducer(datePredictionReducer, {
        result: null,
        loading: false,
        error: null,
    });

    const predict = useCallback(async (date: string) => {
        dispatch({type: "PREDICT_START"});
        try {
            const r = await univariateApi.predictDate(date);
            dispatch({type: "PREDICT_DONE", result: r.data});
        } catch (e: unknown) {
            // isAxiosError narrows the unknown error to AxiosError — no any needed
            const message = axios.isAxiosError(e)
                ? (e.response?.data?.detail ?? "Prediction failed. Please try again.")
                : "Prediction failed. Please try again.";
            dispatch({type: "PREDICT_ERROR", error: message});
        }
    }, []);

    const reset = useCallback(() => {
        dispatch({type: "RESET"});
    }, []);

    return {...state, predict, reset};
}

// ── useLongRangeForecast ──────────────────────────────────────────────────────

interface LongRangeState {
    data: UnivariateRangePoint[];
    loading: boolean;
    error: string | null;
}

type LongRangeAction =
    | { type: "LOAD_START" }
    | { type: "LOAD_DONE"; data: UnivariateRangePoint[] }
    | { type: "LOAD_ERROR"; error: string };

function longRangeReducer(state: LongRangeState, action: LongRangeAction): LongRangeState {
    switch (action.type) {
        case "LOAD_START":
            return {...state, loading: true, error: null};
        case "LOAD_DONE":
            return {data: action.data, loading: false, error: null};
        case "LOAD_ERROR":
            return {...state, loading: false, error: action.error};
    }
}

export function useLongRangeForecast() {
    const [state, dispatch] = useReducer(longRangeReducer, {
        data: [],
        loading: false,
        error: null,
    });

    const load = useCallback(async (startDate: string, endDate: string) => {
        dispatch({type: "LOAD_START"});
        try {
            const r = await univariateApi.predictRange(startDate, endDate);
            dispatch({type: "LOAD_DONE", data: r.data});
        } catch (e: unknown) {
            const message = axios.isAxiosError(e)
                ? (e.response?.data?.detail ?? "Failed to load forecast range.")
                : "Failed to load forecast range.";
            dispatch({type: "LOAD_ERROR", error: message});
        }
    }, []);

    return {...state, load};
}