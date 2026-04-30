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
        // <div className="space-y-4">
        //     <div className="flex items-start gap-4 p-4 bg-emerald-950/30 rounded-xl border border-emerald-800/50">
        //         <div className="w-3 h-3 rounded-full bg-emerald-400 mt-0.5 shrink-0 animate-pulse"/>
        //         <div className="flex-1">
        //             <p className="text-sm font-semibold text-emerald-400">Drift report available</p>
        //             {drift.size_kb && (
        //                 <p className="text-xs text-gray-400 mt-0.5">Size: {drift.size_kb} KB ·
        //                     Source: {drift.source}</p>
        //             )}
        //             <a
        //                 href={reportUrl}
        //                 target="_blank"
        //                 rel="noopener noreferrer"
        //                 className="text-xs text-blue-400 hover:text-blue-300 underline mt-1 inline-block"
        //             >
        //                 Open full report ↗
        //             </a>
        //         </div>
        //     </div>

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