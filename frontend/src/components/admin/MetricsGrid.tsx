import type { Metrics } from "@/types";
import { StatCard } from "@/components/ui/StatCard";

export function MetricsGrid({ metrics }: { metrics: Metrics }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
      <StatCard label="MAE" value={metrics.mae} sub="Mean absolute error" accent="text-blue-400" />
      <StatCard label="MSE" value={metrics.mse} sub="Mean squared error" accent="text-purple-400" />
      <StatCard label="RMSE" value={metrics.rmse} sub="Root mean squared error" accent="text-cyan-400" />
      <StatCard label="Mean Error" value={metrics.mean_error} sub="Average error per day" accent="text-amber-400" />
      <StatCard label="Max Error" value={metrics.max_error} sub="Worst single prediction" accent="text-red-400" />
    </div>
  );
}