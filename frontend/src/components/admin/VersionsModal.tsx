"use client";

import {Modal} from "@/components/ui/Modal";
import {InlineSpinner} from "@/components/ui/LoadingSpinner";
import type {ModelView} from "@/hooks/useAdmin";

interface VersionDetail {
    version: string;
    stage: string;
    run_id: string;
    status: string;
    creation_timestamp: number;
    metrics: Record<string, number>;
    params: Record<string, string>;
}

interface Props {
    open: boolean;
    onClose: () => void;
    modelName: string;
    modelLabel: string;
    versions: VersionDetail[];
    loading: boolean;
    activeVersion: string;
    activating: string | null;
    onActivate: (version: string) => void;
}

function formatDate(ts: number) {
    return new Date(ts).toLocaleDateString("en-GB", {
        day: "numeric", month: "short", year: "numeric",
    });
}

const KEY_METRICS = ["test_mae", "test_rmse", "full_mae", "full_rmse"];

export function VersionsModal({
                                  open, onClose, modelName, modelLabel,
                                  versions, loading, activeVersion, activating, onActivate,
                              }: Props) {
    return (
        <Modal
            open={open}
            onClose={onClose}
            title={modelName}
            subtitle={modelLabel}
            width="max-w-5xl"
        >
            {loading ? (
                <div className="flex items-center justify-center py-16">
                    <div className="flex flex-col items-center gap-3">
                        <div
                            className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"/>
                        <p className="text-sm text-gray-400">Loading versions...</p>
                    </div>
                </div>
            ) : versions.length === 0 ? (
                <p className="text-gray-500 text-sm py-12 text-center">No versions found for this model.</p>
            ) : (
                <table className="w-full text-sm">
                    <thead>
                    <tr className="border-b border-gray-800">
                        {["Version", "Status", "Stage", "Created", ...KEY_METRICS, "Action"].map((h) => (
                            <th
                                key={h}
                                className="text-left text-xs font-semibold text-gray-500 uppercase tracking-wider pb-3 pr-4 whitespace-nowrap"
                            >
                                {h}
                            </th>
                        ))}
                    </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/60">
                    {versions.map((v) => {
                        const isActive =
                            activeVersion === v.version ||
                            (activeVersion === "latest" && v.version === versions[0]?.version);
                        const isActivating = activating === v.version;

                        return (
                            <tr
                                key={v.version}
                                className={`transition-colors ${isActive ? "bg-emerald-950/20" : "hover:bg-gray-800/30"}`}
                            >
                                {/* Version */}
                                <td className="py-3 pr-4">
                                    <div className="flex items-center gap-2">
                                        {isActive && (
                                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0"/>
                                        )}
                                        <span className="font-mono text-white font-semibold">v{v.version}</span>
                                        {isActive && (
                                            <span
                                                className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400">
                          Active
                        </span>
                                        )}
                                    </div>
                                </td>

                                {/* Status */}
                                <td className="py-3 pr-4">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        v.status === "READY"
                            ? "bg-emerald-950 text-emerald-400"
                            : "bg-gray-800 text-gray-400"
                    }`}>
                      {v.status}
                    </span>
                                </td>

                                {/* Stage */}
                                <td className="py-3 pr-4">
                                    {v.stage && v.stage !== "None" ? (
                                        <span
                                            className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-950 text-purple-400">
                        {v.stage}
                      </span>
                                    ) : (
                                        <span className="text-xs text-gray-600">—</span>
                                    )}
                                </td>

                                {/* Created */}
                                <td className="py-3 pr-4 text-xs text-gray-400 whitespace-nowrap">
                                    {formatDate(v.creation_timestamp)}
                                </td>

                                {/* Metrics */}
                                {KEY_METRICS.map((k) => (
                                    <td key={k} className="py-3 pr-4">
                                        {v.metrics[k] !== undefined ? (
                                            <span className="font-mono text-xs text-white">{v.metrics[k]}</span>
                                        ) : (
                                            <span className="text-xs text-gray-600">—</span>
                                        )}
                                    </td>
                                ))}

                                {/* Action */}
                                <td className="py-3">
                                    {isActive ? (
                                        <span className="text-xs text-gray-600">Current</span>
                                    ) : (
                                        <button
                                            onClick={() => onActivate(v.version)}
                                            disabled={!!activating}
                                            className="text-xs px-3 py-1.5 border border-emerald-700 text-emerald-400 rounded-lg hover:bg-emerald-950 disabled:opacity-40 transition-colors flex items-center gap-1.5 whitespace-nowrap"
                                        >
                                            {isActivating ? <InlineSpinner/> : null}
                                            Activate
                                        </button>
                                    )}
                                </td>
                            </tr>
                        );
                    })}
                    </tbody>
                </table>
            )}
        </Modal>
    );
}