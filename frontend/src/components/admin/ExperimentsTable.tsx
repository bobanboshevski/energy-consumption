import type { ExperimentRun } from "@/types";
import { StatusBadge } from "@/components/ui/Badge";

function formatTime(ts: number) {
  return new Date(ts).toLocaleString("en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"
  });
}

export function ExperimentsTable({ runs }: { runs: ExperimentRun[] }) {
  if (runs.length === 0) {
    return <p className="text-gray-500 text-sm py-8 text-center">No experiment runs found.</p>;
  }

  return (
    <div className="space-y-3">
      {runs.map((r) => (
        <div key={r.run_id} className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
          <div className="flex items-start justify-between mb-3">
            <div>
              <p className="font-medium text-white">{r.run_name}</p>
              <p className="text-xs text-gray-500 font-mono mt-0.5">{r.run_id.slice(0, 16)}...</p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500">{formatTime(r.start_time)}</span>
              <StatusBadge status={r.status} />
            </div>
          </div>
          {Object.keys(r.metrics).length > 0 && (
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mt-3">
              {Object.entries(r.metrics).map(([k, v]) => (
                <div key={k} className="bg-gray-900 rounded-lg p-2.5">
                  <p className="text-xs text-gray-500 mb-1">{k}</p>
                  <p className="font-mono text-xs text-white font-semibold">{Number(v).toFixed(4)}</p>
                </div>
              ))}
            </div>
          )}
          {Object.keys(r.params).length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              {Object.entries(r.params).slice(0, 5).map(([k, v]) => (
                <span key={k} className="text-xs bg-gray-900 text-gray-400 px-2.5 py-1 rounded-lg font-mono">
                  {k}: {v}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}