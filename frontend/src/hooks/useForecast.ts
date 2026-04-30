import { useState, useEffect } from "react";
import { predictionsApi } from "@/lib/api";
import type { ForecastPoint, HistoricalPoint } from "@/types";

export function useForecast() {
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [historical, setHistorical] = useState<HistoricalPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async (days = 60) => {
    setLoading(true);
    setError(null);
    try {
      const [f, h] = await Promise.all([
        predictionsApi.getForecast(),
        predictionsApi.getHistorical(days),
      ]);
      setForecast(f.data);
      setHistorical(h.data);
    } catch (e) {
      setError("Failed to load data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const refresh = async () => {
    await predictionsApi.refreshData();
    await load();
  };

  useEffect(() => { load(); }, []);

  return { forecast, historical, loading, error, refresh };
}