import type {DriftReport} from "@/types";
import {monitoringApi} from "@/lib/api";

export function DriftStatus({drift}: { drift: DriftReport }) {
    const reportUrl = monitoringApi.getDriftReportUrl();

    if (!drift.available) {
        return (
            <div className="flex items-center gap-3 p-4 bg-gray-800/50 rounded-xl border border-gray-700">
                <div className="w-3 h-3 rounded-full bg-gray-600"/>
                <div>
                    <p className="text-sm font-medium text-gray-300">No drift report available</p>
                    <p className="text-xs text-gray-500 mt-0.5">{drift.reason || "Report not generated yet"}</p>
                </div>
            </div>
        );
    }

    return (
            <div className="rounded-xl overflow-hidden border border-gray-700" style={{height: 600}}>
                <iframe
                    src={reportUrl}
                    className="w-full h-full bg-white"
                    title="Evidently Drift Report"
                />
            </div>
        // </div>
    )
        ;
}