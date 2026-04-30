import type { ForecastPoint } from "@/types";
import { DemandBadge } from "@/components/ui/Badge";

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export function ForecastTable({ forecast }: { forecast: ForecastPoint[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800">
            <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-3 pr-4">Date</th>
            <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-3 pr-4">Demand (GW)</th>
            <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-3 pr-4">Category</th>
            <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-3 pr-4">Temp Range</th>
            <th className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-3">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800/50">
          {forecast.map((f) => (
            <tr key={f.date} className="hover:bg-gray-800/30 transition-colors">
              <td className="py-3 pr-4 text-gray-200 font-medium whitespace-nowrap">{formatDate(f.date)}</td>
              <td className="py-3 pr-4 font-mono text-white font-semibold">{f.predicted_demand.toFixed(4)}</td>
              <td className="py-3 pr-4"><DemandBadge category={f.demand_category} /></td>
              <td className="py-3 pr-4 text-gray-400 whitespace-nowrap">
                {f.temp_min !== null && f.temp_max !== null
                  ? `${f.temp_min}°C – ${f.temp_max}°C`
                  : "—"}
              </td>
              <td className="py-3">
                {f.is_confirmed ? (
                  <span className="text-xs text-amber-400 bg-amber-950 px-2.5 py-1 rounded-full font-medium">
                    Awaiting data
                  </span>
                ) : (
                  <span className="text-xs text-blue-400 bg-blue-950 px-2.5 py-1 rounded-full font-medium">
                    Forecast
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}