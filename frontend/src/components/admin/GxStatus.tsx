import type {DriftReport, ValidationReport} from "@/types";
import {monitoringApi} from "@/lib/api";

export function GxStatus({report}: { report: ValidationReport }) {
    const reportUrl = monitoringApi.getGxReportUrl();

    if (!report.available) {
        return (
            <div className="flex items-center gap-3 p-4 bg-gray-800/50 rounded-xl border border-gray-700">
                <div className="w-3 h-3 rounded-full bg-gray-600"/>
                <div>
                    <p className="text-sm font-medium text-gray-300">No GX validation report available</p>
                    <p className="text-xs text-gray-500 mt-0.5">{report.reason || "Report not generated yet"}</p>
                </div>
            </div>
        );
    }

    const passed = (report as any).passed !== false;

    return (
        <div className="space-y-4">

            <div className={`flex items-start gap-4 p-4 rounded-xl border ${
                passed
                    ? "bg-emerald-950/30 border-emerald-800/50"
                    : "bg-red-950/30 border-red-800/50"
            }`}>
                <div className={`w-3 h-3 rounded-full mt-0.5 shrink-0 animate-pulse ${
                    passed ? "bg-emerald-400" : "bg-red-400"
                }`}/>
                <div className="flex-1">
                    <p className={`text-sm font-semibold ${passed ? "text-emerald-400" : "text-red-400"}`}>
                        {passed ? "Validation passed" : "Validation failed"}
                    </p>
                    {report.size_kb && (
                        <p className="text-xs text-gray-400 mt-0.5">
                            Size: {report.size_kb} KB · Source: {(report as any).source}
                        </p>
                    )}

                    <a href={reportUrl}
                       target="_blank"
                       rel="noopener noreferrer"
                       className="text-xs text-blue-400 hover:text-blue-300 underline mt-1 inline-block"
                    >
                        Open full report ↗
                    </a>
                </div>
            </div>

            <div className="rounded-xl overflow-hidden border border-gray-700" style={{height: 600}}>
                <iframe
                    src={reportUrl}
                    className="w-full h-full"
                    title="GX Validation Report"
                />
            </div>
        </div>

    );
}