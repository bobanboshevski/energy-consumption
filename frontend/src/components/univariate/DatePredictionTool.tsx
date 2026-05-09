"use client";

import {useState} from "react";
import {useDatePrediction, useUnivariateInfo} from "@/hooks/useUnivariate";
import {DemandBadge} from "@/components/ui/Badge";
import {InlineSpinner} from "@/components/ui/LoadingSpinner";

function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleDateString("en-GB", {
        weekday: "long",
        day: "numeric",
        month: "long",
        year: "numeric",
    });
}

export function DatePredictionTool() {
    const {info} = useUnivariateInfo();
    const {result, loading, error, predict, reset} = useDatePrediction();

    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    const maxDate = info?.max_date
        || new Date(today.getTime() + 365 * 86400000).toISOString().split("T")[0];

    const minDate = tomorrow.toISOString().split("T")[0];

    const [selectedDate, setSelectedDate] = useState<string>("");

    const handlePredict = () => {
        if (selectedDate) predict(selectedDate);
    };

    const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setSelectedDate(e.target.value);
        reset();
    };

    return (
        <div className="space-y-5">
            {/* bg-blue-950/30 border border-blue-800/40 rounded-xl */}
            <div className="flex items-center gap-2 p-3 ">
                <span className="text-blue-400 text-sm">🔮</span>
                <p className="text-xs text-blue-300">
                    Univariate model — uses only historical demand patterns.
                    Predict up to <span className="font-semibold">365 days</span> ahead without weather data.
                </p>
            </div>

            <div className="flex gap-3 items-end">
                <div className="flex-1">
                    <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                        Select date
                    </label>
                    <input
                        type="date"
                        value={selectedDate}
                        min={minDate}
                        max={maxDate}
                        onChange={handleDateChange}
                        className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-white text-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                       [color-scheme:dark]"
                    />
                </div>
                <div className="flex flex-col">
                    <span className="block text-xs mb-2 invisible">placeholder</span>

                    <button
                        onClick={handlePredict}
                        disabled={!selectedDate || loading}
                        className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed
                     text-white text-sm font-medium rounded-xl transition-colors flex items-center gap-2 whitespace-nowrap"
                    >
                        {loading ? <><InlineSpinner/> Predicting...</> : "Predict demand"}
                    </button>
                </div>

            </div>

            {error && (
                <div className="p-4 bg-red-950/40 border border-red-800/50 rounded-xl">
                    <p className="text-sm text-red-400">{error}</p>
                </div>
            )}

            {result && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-5 space-y-4">
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Prediction for</p>
                            <p className="text-lg font-semibold text-white">{formatDate(result.target_date)}</p>
                            <p className="text-xs text-gray-500 mt-0.5">{result.days_ahead} days from today</p>
                        </div>
                        <DemandBadge category={result.demand_category}/>
                    </div>

                    <div className="flex items-end gap-2">
                        <span className="text-5xl font-bold font-mono text-white">
                          {result.predicted_demand.toFixed(3)}
                        </span>
                        <span className="text-xl text-gray-400 mb-1">GW</span>
                    </div>

                    <div className="pt-3 border-t border-gray-700 grid grid-cols-2 gap-3">
                        {/*<div className="bg-gray-900 rounded-xl p-3">*/}
                        {/*    <p className="text-xs text-gray-500 mb-1">Model</p>*/}
                        {/*    <p className="text-xs text-gray-300 font-mono">{result.model}</p>*/}
                        {/*</div>*/}
                        <div className="bg-gray-900 rounded-xl p-3">
                            <p className="text-xs text-gray-500 mb-1">Last known data</p>
                            <p className="text-xs text-gray-300 font-mono">{result.last_known_date}</p>
                        </div>
                    </div>

                    <p className="text-xs text-gray-600 italic">{result.note}</p>
                </div>
            )}

            {!result && !error && !loading && (
                <div className="text-center py-8 text-gray-600 text-sm">
                    Select a date above to predict energy demand
                </div>
            )}
        </div>
    );
}