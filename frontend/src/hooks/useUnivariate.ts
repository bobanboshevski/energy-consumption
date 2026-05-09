import {useState, useEffect} from "react";
import {univariateApi} from "@/lib/api";
import type {UnivariatePrediction, UnivariateRangePoint, UnivariateModelInfo} from "@/types";

export function useUnivariateInfo() {
    const [info, setInfo] = useState<UnivariateModelInfo | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        univariateApi.getInfo()
            .then((r) => setInfo(r.data))
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    return {info, loading};
}

export function useDatePrediction() {
    const [result, setResult] = useState<UnivariatePrediction | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const predict = async (date: string) => {
        setLoading(true);
        setError(null);
        setResult(null);
        try {
            const r = await univariateApi.predictDate(date);
            setResult(r.data);
        } catch (e: any) {
            setError(e.response?.data?.detail || "Prediction failed. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const reset = () => {
        setResult(null);
        setError(null);
    };

    return {result, loading, error, predict, reset};
}

export function useLongRangeForecast() {
    const [data, setData] = useState<UnivariateRangePoint[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = async (startDate: string, endDate: string) => {
        setLoading(true);
        setError(null);
        try {
            const r = await univariateApi.predictRange(startDate, endDate);
            setData(r.data);
        } catch (e: any) {
            setError(e.response?.data?.detail || "Failed to load forecast range.");
        } finally {
            setLoading(false);
        }
    };

    return {data, loading, error, load};
}