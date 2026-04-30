"use client";

import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts";
import type { HistoricalPoint, ForecastPoint } from "@/types";

interface ChartRow {
  date: string;
  actual?: number;
  forecast?: number;
}

function formatDate(d: string) {
  return new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-950 border border-gray-700 rounded-xl px-4 py-3 shadow-xl">
      <p className="text-xs text-gray-400 mb-2">{formatDate(label)}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-gray-300">{p.name}:</span>
          <span className="font-mono font-semibold text-white">{Number(p.value).toFixed(4)} GW</span>
        </div>
      ))}
    </div>
  );
};

interface Props {
  historical: HistoricalPoint[];
  forecast: ForecastPoint[];
  todayDate?: string;
}

export function ForecastChart({ historical, forecast, todayDate }: Props) {
  const historicalRows: ChartRow[] = historical.map((h) => ({
    date: h.Date,
    actual: h.energy_demand,
  }));

  const forecastRows: ChartRow[] = forecast.map((f) => ({
    date: f.date,
    forecast: f.predicted_demand,
  }));

  // Overlap last historical point with first forecast for visual continuity
//   if (historicalRows.length > 0 && forecastRows.length > 0) {
//     const last = historicalRows[historicalRows.length - 1];
//     forecastRows[0] = { ...forecastRows[0], actual: last.actual };
//   }

  const data = [...historicalRows, ...forecastRows];

  return (
    <ResponsiveContainer width="100%" height={380}>
      <ComposedChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 20 }}>
        <defs>
          <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f97316" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickFormatter={formatDate}
          interval={Math.floor(data.length / 8)}
          label={{ value: "Date", position: "insideBottom", offset: -10, fill: "#6b7280", fontSize: 12 }}
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickFormatter={(v) => `${v.toFixed(1)}`}
          domain={["auto", "auto"]}
          label={{ value: "Gigawatts (GW)", angle: -90, position: "insideLeft", offset: 15, fill: "#6b7280", fontSize: 12 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: "13px", color: "#9ca3af", paddingTop: "16px" }}
        />
        {todayDate && (
          <ReferenceLine
            x={todayDate}
            stroke="#4b5563"
            strokeDasharray="6 3"
            label={{
                value: "Today",
                position: "insideTopLeft",
                fill: "#6b7280",
                fontSize: 11,
                dy: 8
            }}
          />
        )}
        <Area
          type="monotone"
          dataKey="actual"
          fill="url(#actualGrad)"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="Actual"
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="forecast"
          stroke="#f97316"
          strokeWidth={2}
          strokeDasharray="6 3"
          dot={false}
          name="Forecast"
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}