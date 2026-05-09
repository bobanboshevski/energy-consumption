// "use client";
//
// import { useEffect, useState } from "react";
// import { getForecast, getHistorical } from "@/lib/api";
// import { ForecastPoint, HistoricalPoint } from "@/types";
// import {
//   LineChart, Line, XAxis, YAxis, CartesianGrid,
//   Tooltip, Legend, ResponsiveContainer, ReferenceLine
// } from "recharts";
//
// const categoryColors = {
//   low: "#22c55e",
//   medium: "#f59e0b",
//   high: "#ef4444",
// };
//
// export default function Home() {
//   const [forecast, setForecast] = useState<ForecastPoint[]>([]);
//   const [historical, setHistorical] = useState<HistoricalPoint[]>([]);
//   const [loading, setLoading] = useState(true);
//
//   useEffect(() => {
//     Promise.all([getForecast(), getHistorical(60)])
//       .then(([f, h]) => {
//         setForecast(f.data);
//         setHistorical(h.data);
//       })
//       .finally(() => setLoading(false));
//   }, []);
//
//   const chartData = [
//     ...historical.slice(-30).map((h) => ({
//       date: h.Date,
//       actual: h.energy_demand,
//       predicted: null,
//       type: "historical",
//     })),
//     ...forecast.map((f) => ({
//       date: f.date,
//       actual: null,
//       predicted: f.predicted_demand,
//       category: f.demand_category,
//       type: "forecast",
//     })),
//   ];
//
//   if (loading) return (
//     <div className="flex items-center justify-center h-screen">
//       <div className="text-lg text-gray-500">Loading energy forecast...</div>
//     </div>
//   );
//
//   return (
//     <main className="max-w-6xl mx-auto px-4 py-8">
//       <h1 className="text-3xl font-bold mb-2">⚡ Energy Demand Forecast</h1>
//       <p className="text-gray-500 mb-8">Slovenia — next {forecast.length} days prediction</p>
//
//       <div className="grid grid-cols-3 gap-4 mb-8">
//         {forecast.slice(0, 3).map((f) => (
//           <div key={f.date} className="bg-white rounded-xl border p-4 shadow-sm">
//             <div className="text-sm text-gray-500">{f.date}</div>
//             <div className="text-2xl font-bold mt-1">{f.predicted_demand.toFixed(3)} GW</div>
//             <div
//               className="text-sm font-medium mt-1 capitalize"
//               style={{ color: categoryColors[f.demand_category] }}
//             >
//               {f.demand_category} demand
//             </div>
//             <div className="text-xs text-gray-400 mt-2">
//               {f.temp_min}°C — {f.temp_max}°C
//             </div>
//           </div>
//         ))}
//       </div>
//
//       <div className="bg-white rounded-xl border p-6 shadow-sm mb-8">
//         <h2 className="text-lg font-semibold mb-4">Historical + Forecast</h2>
//         <ResponsiveContainer width="100%" height={320}>
//           <LineChart data={chartData}>
//             <CartesianGrid strokeDasharray="3 3" />
//             <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={6} />
//             <YAxis tick={{ fontSize: 11 }} />
//             <Tooltip />
//             <Legend />
//             <ReferenceLine x={historical[historical.length - 1]?.Date} stroke="#94a3b8" strokeDasharray="4 4" label="Today" />
//             <Line type="monotone" dataKey="actual" stroke="#3b82f6" dot={false} name="Actual" />
//             <Line type="monotone" dataKey="predicted" stroke="#f97316" dot={false} strokeDasharray="5 5" name="Forecast" />
//           </LineChart>
//         </ResponsiveContainer>
//       </div>
//
//       <div className="bg-white rounded-xl border p-6 shadow-sm">
//         <h2 className="text-lg font-semibold mb-4">Upcoming Forecast</h2>
//         <table className="w-full text-sm">
//           <thead>
//             <tr className="text-left text-gray-500 border-b">
//               <th className="pb-2">Date</th>
//               <th className="pb-2">Demand (GW)</th>
//               <th className="pb-2">Category</th>
//               <th className="pb-2">Temp Range</th>
//             </tr>
//           </thead>
//           <tbody>
//             {forecast.map((f) => (
//               <tr key={f.date} className="border-b last:border-0">
//                 <td className="py-2">{f.date}</td>
//                 <td className="py-2 font-mono">{f.predicted_demand.toFixed(4)}</td>
//                 <td className="py-2">
//                   <span
//                     className="px-2 py-0.5 rounded-full text-xs font-medium capitalize"
//                     style={{
//                       backgroundColor: categoryColors[f.demand_category] + "22",
//                       color: categoryColors[f.demand_category],
//                     }}
//                   >
//                     {f.demand_category}
//                   </span>
//                 </td>
//                 <td className="py-2 text-gray-500">{f.temp_min}°C – {f.temp_max}°C</td>
//               </tr>
//             ))}
//           </tbody>
//         </table>
//       </div>
//     </main>
//   );
// }

"use client";

import {useForecast} from "@/hooks/useForecast";
import {ForecastCards} from "@/components/forecast/ForecastCards";
import {ForecastChart} from "@/components/forecast/ForecastChart";
import {ForecastTable} from "@/components/forecast/ForecastTable";
import {LoadingSpinner} from "@/components/ui/LoadingSpinner";
import {Card, CardHeader} from "@/components/ui/Card";
import {DatePredictionTool} from "@/components/univariate/DatePredictionTool";
import {LongRangeForecastChart} from "@/components/univariate/LongRangeForecastChart";

export default function HomePage() {
    const {forecast, historical, loading, error, refresh} = useForecast();

//   const todayDate = historical.length > 0 ? historical[historical.length - 1].Date : undefined;
//   const todayDate = forecast.length > 0 ? forecast[0].date : undefined;

    const todayDate = new Date().toISOString().split("T")[0]; // "YYYY-MM-DD"

    if (loading) return <LoadingSpinner text="Loading energy forecast..."/>;

    if (error) return (
        <div className="flex items-center justify-center h-64">
            <div className="text-center">
                <p className="text-red-400 mb-3">{error}</p>
                <button onClick={refresh} className="text-sm text-blue-400 hover:text-blue-300 underline">
                    Try again
                </button>
            </div>
        </div>
    );

    return (
        <div className="max-w-7xl mx-auto px-6 py-8">

            {/* ── Header ──────────────────────────────────────────────── */}
            <div className="flex items-start justify-between mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-white">Energy Demand Forecast</h1>
                    <p className="text-gray-400 mt-1">
                        Slovenia · {forecast.length} days ahead · Updated daily
                    </p>
                </div>
                <button
                    onClick={refresh}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-xl text-sm text-gray-300 transition-colors"
                >
                    ↻ Refresh
                </button>
            </div>

            {/* ── Next 3 days cards ───────────────────────────────────── */}
            <ForecastCards forecast={forecast}/>

            <Card className="mb-6 p-6">
                <CardHeader
                    title="Historical & Forecast"
                    subtitle={`Last ${historical.length} days of actual data + ${forecast.length} day forecast`}
                />
                <ForecastChart historical={historical} forecast={forecast} todayDate={todayDate}/>
            </Card>

            {/* ── Short-range forecast table ───────────────────────────── */}
            <Card>
                <CardHeader
                    title="Detailed Forecast"
                    subtitle="Day-by-day predictions with weather context"
                />
                <div className="px-6 pb-6">
                    <ForecastTable forecast={forecast}/>
                </div>
            </Card>

            {/* ── Long-range section divider ───────────────────────────── */}
            <div className="flex items-center gap-4 py-2 mt-2 mb-2">
                <div className="flex-1 h-px bg-gray-800"/>
                <span className="text-s text-gray-600 uppercase tracking-widest font-medium">
                        Long-range prediction · up to 365 days
                    </span>
                <div className="flex-1 h-px bg-gray-800"/>
            </div>

            {/* ── Two-column: date picker + long-range chart ──────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
                <Card className="p-6">
                    <CardHeader
                        title="Predict Any Date"
                        subtitle="Select a specific date up to 365 days ahead"
                    />
                    <div className="px-6 pb-6">
                        <DatePredictionTool/>
                    </div>
                </Card>

                <Card className="p-6">
                    <CardHeader
                        title="Long-range Outlook"
                        subtitle="Energy demand trend over the coming months"
                    />
                    <div className="px-6 pb-6">
                        <LongRangeForecastChart/>
                    </div>
                </Card>
            </div>

        </div>
    );
}