import type {ForecastPoint} from "@/types";
import {DemandBadge} from "@/components/ui/Badge";

function dayLabel(dateStr: string) {
    const d = new Date(dateStr);
    return d.toLocaleDateString("en-GB", {weekday: "short", day: "numeric", month: "short"});
}

export function ForecastCards({forecast}: { forecast: ForecastPoint[] }) {

    const todayStr = new Date().toISOString().split("T")[0];

    const todayIndex = forecast.findIndex(f => f.date === todayStr);
    const startIndex = todayIndex === -1 ? 0 : todayIndex;
    const next3 = forecast.slice(startIndex, startIndex + 3);

    // const next3 = forecast.slice(0, 3);

    return (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            {next3.map((f, idx) => (
                <div
                    key={f.date}
                    className={`relative bg-gray-900 border rounded-2xl p-5 overflow-hidden transition-all hover:border-blue-700 ${
                        idx === 0 ? "border-blue-600" : "border-gray-800"
                    }`}
                >
                    {idx === 0 && (
                        <span
                            className="absolute top-3 right-3 text-xs bg-blue-600 text-white px-2 py-0.5 rounded-full font-medium">
              Today
            </span>
                    )}
                    {idx === 1 && (
                        <span
                            className="absolute top-3 right-3 text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded-full font-medium">
                Tomorrow
            </span>
                    )}
                    <p className="text-xs text-gray-500 font-medium mb-3">{dayLabel(f.date)}</p>
                    <p className="text-4xl font-bold font-mono text-white mb-1">
                        {f.predicted_demand.toFixed(3)}
                        <span className="text-lg text-gray-400 ml-1">GW</span>
                    </p>
                    <div className="mt-3 mb-4">
                        <DemandBadge category={f.demand_category}/>
                        {f.is_confirmed && (
                            <span className="ml-2 text-xs text-gray-500 italic">data pending</span>
                        )}
                    </div>
                    <div className="flex items-center gap-3 text-sm text-gray-400 border-t border-gray-800 pt-3">
                        <span>🌡 {f.temp_min}° – {f.temp_max}°C</span>
                    </div>
                </div>
            ))}
        </div>
    );
}