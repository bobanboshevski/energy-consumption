// "use client";
//
// import { useState, useEffect } from "react";
// import type { RegisteredModel } from "@/types";
// import { InlineSpinner } from "@/components/ui/LoadingSpinner";
// import { modelsApi } from "@/lib/api";
//
// interface VersionDetail {
//   version: string;
//   stage: string;
//   run_id: string;
//   status: string;
//   creation_timestamp: number;
//   metrics: Record<string, number>;
//   params: Record<string, string>;
// }
//
// interface Props {
//   models: RegisteredModel[];
// }
//
// export function ModelRegistry({ models }: Props) {
//   const [loading, setLoading] = useState<string | null>(null);
//   const [activeVersion, setActiveVersion] = useState<string>("latest");
//   const [loadedVersion, setLoadedVersion] = useState<string>("unknown");
//   const [details, setDetails] = useState<Record<string, VersionDetail[]>>({});
//
//   useEffect(() => {
//     modelsApi.getActive().then((r) => {
//       setActiveVersion(r.data.active_version);
//       setLoadedVersion(r.data.loaded_version);
//     });
//     models.forEach(async (m) => {
//       try {
//         const res = await modelsApi.getVersions(m.name);
//         setDetails((prev) => ({ ...prev, [m.name]: res.data }));
//       } catch (e) {
//         console.error(`Failed to load versions for ${m.name}`, e);
//       }
//     });
//   }, [models]);
//
//   const handleActivate = async (version: string) => {
//     setLoading(version);
//     try {
//       await modelsApi.activate(version);
//       const active = await modelsApi.getActive();
//       setActiveVersion(active.data.active_version);
//       setLoadedVersion(active.data.loaded_version);
//     } finally {
//       setLoading(null);
//     }
//   };
//
//   return (
//     <div className="space-y-6">
//       <div className="bg-blue-950/30 border border-blue-800/50 rounded-xl p-4 flex items-center justify-between">
//         <div>
//           <p className="text-sm text-blue-400 font-medium">Active model</p>
//           <p className="text-xs text-gray-400 mt-0.5">
//             Config: <span className="font-mono text-white">v{activeVersion}</span> · Loaded:{" "}
//             <span className="font-mono text-white">v{loadedVersion}</span>
//           </p>
//         </div>
//         <button
//           onClick={() => handleActivate("latest")}
//           disabled={activeVersion === "latest" || !!loading}
//           className="text-xs px-3 py-1.5 border border-blue-700 text-blue-400 rounded-lg hover:bg-blue-950 disabled:opacity-40 transition-colors"
//         >
//           Reset to latest
//         </button>
//       </div>
//
//       {models.length === 0 && (
//         <p className="text-gray-500 text-sm py-8 text-center">No registered models found.</p>
//       )}
//       {models.map((m) => {
//         const versionDetails = details[m.name] || [];
//         return (
//           <div key={m.name} className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
//             <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-700">
//               <div className="w-8 h-8 bg-blue-600/20 rounded-lg flex items-center justify-center">
//                 <span className="text-blue-400 text-xs font-bold">M</span>
//               </div>
//               <div>
//                 <h3 className="font-semibold text-white">{m.name}</h3>
//                 <p className="text-xs text-gray-500">
//                   {versionDetails.length} version{versionDetails.length !== 1 ? "s" : ""}
//                 </p>
//               </div>
//             </div>
//
//             <div className="divide-y divide-gray-700/50">
//               {versionDetails.map((v) => {
//                 const isActive =
//                   activeVersion === v.version ||
//                   (activeVersion === "latest" && v.version === versionDetails[0]?.version);
//                 const isLoading = loading === v.version;
//
//                 return (
//                   <div
//                     key={v.version}
//                     className={`px-5 py-4 ${isActive ? "bg-emerald-950/20 border-l-2 border-emerald-500" : ""}`}
//                   >
//                     <div className="flex items-start justify-between mb-3">
//                       <div className="flex items-center gap-3">
//                         <span className="font-mono text-sm text-white font-semibold">v{v.version}</span>
//                         {isActive && (
//                           <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400">
//                             Active
//                           </span>
//                         )}
//                         <span
//                           className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
//                             v.status === "READY"
//                               ? "bg-emerald-950 text-emerald-400"
//                               : "bg-gray-800 text-gray-400"
//                           }`}
//                         >
//                           {v.status}
//                         </span>
//                         <span className="text-xs text-gray-500">
//                           {new Date(v.creation_timestamp).toLocaleDateString("en-GB", {
//                             day: "numeric",
//                             month: "short",
//                             year: "numeric",
//                           })}
//                         </span>
//                       </div>
//                       {!isActive && (
//                         <button
//                           onClick={() => handleActivate(v.version)}
//                           disabled={!!loading}
//                           className="text-xs px-3 py-1.5 border border-emerald-700 text-emerald-400 rounded-lg hover:bg-emerald-950 disabled:opacity-40 transition-colors flex items-center gap-1.5"
//                         >
//                           {isLoading ? <InlineSpinner /> : null}
//                           Activate
//                         </button>
//                       )}
//                     </div>
//
//                     {Object.keys(v.metrics).length > 0 && (
//                       <div className="grid grid-cols-4 gap-2 mt-2">
//                         {Object.entries(v.metrics).map(([k, val]) => (
//                           <div key={k} className="bg-gray-900 rounded-lg px-3 py-2">
//                             <p className="text-xs text-gray-500">{k}</p>
//                             <p className="font-mono text-xs text-white font-semibold mt-0.5">{val}</p>
//                           </div>
//                         ))}
//                       </div>
//                     )}
//                   </div>
//                 );
//               })}
//             </div>
//           </div>
//         );
//       })}
//     </div>
//   );
// }


"use client";

import {useState, useEffect} from "react";
import type {RegisteredModel} from "@/types";
import {InlineSpinner} from "@/components/ui/LoadingSpinner";
import {modelsApi} from "@/lib/api";

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
    models: RegisteredModel[];
}

const STAGES = ["Staging", "Production", "Archived"] as const;
type Stage = typeof STAGES[number];

const stageColors: Record<Stage, string> = {
    Staging: "border-yellow-700 text-yellow-400 hover:bg-yellow-950",
    Production: "border-emerald-700 text-emerald-400 hover:bg-emerald-950",
    Archived: "border-gray-600 text-gray-400 hover:bg-gray-800",
};

export function ModelRegistry({models}: Props) {
    const [activating, setActivating] = useState<string | null>(null);
    // const [transitioning, setTransitioning] = useState<string | null>(null);
    const [activeVersion, setActiveVersion] = useState<string>("latest");
    const [loadedVersion, setLoadedVersion] = useState<string>("unknown");
    const [details, setDetails] = useState<Record<string, VersionDetail[]>>({});

    useEffect(() => {
        modelsApi.getActive().then((r) => {
            setActiveVersion(r.data.active_version);
            setLoadedVersion(r.data.loaded_version);
        });
        models.forEach(async (m) => {
            try {
                const res = await modelsApi.getVersions(m.name);
                setDetails((prev) => ({...prev, [m.name]: res.data}));
            } catch (e) {
                console.error(`Failed to load versions for ${m.name}`, e);
            }
        });
    }, [models]);

    const handleActivate = async (version: string) => {
        setActivating(version);
        try {
            await modelsApi.activate(version);
            const active = await modelsApi.getActive();
            setActiveVersion(active.data.active_version);
            setLoadedVersion(active.data.loaded_version);
        } finally {
            setActivating(null);
        }
    };

    return (
        <div className="space-y-6">
            <div className="bg-blue-950/30 border border-blue-800/50 rounded-xl p-4 flex items-center justify-between">
                <div>
                    <p className="text-sm text-blue-400 font-medium">Active model</p>
                    <p className="text-xs text-gray-400 mt-0.5">
                        Config: <span className="font-mono text-white">v{activeVersion}</span> · Loaded:{" "}
                        <span className="font-mono text-white">v{loadedVersion}</span>
                    </p>
                </div>
                <button
                    onClick={() => handleActivate("latest")}
                    disabled={activeVersion === "latest" || !!activating}
                    className="text-xs px-3 py-1.5 border border-blue-700 text-blue-400 rounded-lg hover:bg-blue-950 disabled:opacity-40 transition-colors"
                >
                    Reset to latest
                </button>
            </div>

            {models.length === 0 && (
                <p className="text-gray-500 text-sm py-8 text-center">No registered models found.</p>
            )}

            {models.map((m) => {
                const versionDetails = details[m.name] || [];
                return (
                    <div key={m.name} className="bg-gray-800/40 rounded-xl border border-gray-700 overflow-hidden">
                        <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-700">
                            <div className="w-8 h-8 bg-blue-600/20 rounded-lg flex items-center justify-center">
                                <span className="text-blue-400 text-xs font-bold">M</span>
                            </div>
                            <div>
                                <h3 className="font-semibold text-white">{m.name}</h3>
                                <p className="text-xs text-gray-500">
                                    {versionDetails.length} version{versionDetails.length !== 1 ? "s" : ""}
                                </p>
                            </div>
                        </div>

                        <div className="divide-y divide-gray-700/50">
                            {versionDetails.map((v) => {
                                const isActive =
                                    activeVersion === v.version ||
                                    (activeVersion === "latest" && v.version === versionDetails[0]?.version);
                                const isActivating = activating === v.version;

                                return (
                                    <div
                                        key={v.version}
                                        className={`px-5 py-4 ${isActive ? "bg-emerald-950/20 border-l-2 border-emerald-500" : ""}`}
                                    >
                                        <div className="flex items-start justify-between mb-3">
                                            <div className="flex items-center gap-3 flex-wrap">
                                                <span
                                                    className="font-mono text-sm text-white font-semibold">v{v.version}</span>
                                                {isActive && (
                                                    <span
                                                        className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400">
                            Active
                          </span>
                                                )}
                                                <span
                                                    className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                        v.status === "READY"
                                                            ? "bg-emerald-950 text-emerald-400"
                                                            : "bg-gray-800 text-gray-400"
                                                    }`}
                                                >
                          {v.status}
                        </span>
                                                {v.stage && v.stage !== "None" && (
                                                    <span
                                                        className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-950 text-purple-400">
                            {v.stage}
                          </span>
                                                )}
                                                <span className="text-xs text-gray-500">
                          {new Date(v.creation_timestamp).toLocaleDateString("en-GB", {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                          })}
                        </span>
                                            </div>

                                            <div className="flex items-center gap-2 flex-shrink-0">
                                                {!isActive && (
                                                    <button
                                                        onClick={() => handleActivate(v.version)}
                                                        disabled={!!activating}
                                                        className="text-xs px-3 py-1.5 border border-emerald-700 text-emerald-400 rounded-lg hover:bg-emerald-950 disabled:opacity-40 transition-colors flex items-center gap-1.5"
                                                    >
                                                        {isActivating ? <InlineSpinner/> : null}
                                                        Activate
                                                    </button>
                                                )}
                                            </div>
                                        </div>

                                        {/*/!* Stage transition buttons *!/*/}
                                        {/*<div className="flex flex-wrap gap-2 mb-3">*/}
                                        {/*    {STAGES.map((stage) => {*/}
                                        {/*        const key = `${m.name}-${v.version}-${stage}`;*/}
                                        {/*        const isCurrentStage = v.stage === stage;*/}
                                        {/*        const isTransitioning = transitioning === key;*/}
                                        {/*        return (*/}
                                        {/*            <button*/}
                                        {/*                key={stage}*/}
                                        {/*                onClick={() => handleTransition(m.name, v.version, stage)}*/}
                                        {/*                disabled={isCurrentStage || !!transitioning}*/}
                                        {/*                className={`text-xs px-2.5 py-1 border rounded-lg transition-colors flex items-center gap-1 disabled:opacity-40 ${*/}
                                        {/*                    isCurrentStage*/}
                                        {/*                        ? "border-gray-600 text-gray-500 cursor-default"*/}
                                        {/*                        : stageColors[stage]*/}
                                        {/*                }`}*/}
                                        {/*            >*/}
                                        {/*                {isTransitioning ? <InlineSpinner/> : "→"}*/}
                                        {/*                {stage}*/}
                                        {/*            </button>*/}
                                        {/*        );*/}
                                        {/*    })}*/}
                                        {/*</div>*/}

                                        {Object.keys(v.metrics).length > 0 && (
                                            <div className="grid grid-cols-4 gap-2 mt-2">
                                                {Object.entries(v.metrics).map(([k, val]) => (
                                                    <div key={k} className="bg-gray-900 rounded-lg px-3 py-2">
                                                        <p className="text-xs text-gray-500">{k}</p>
                                                        <p className="font-mono text-xs text-white font-semibold mt-0.5">{val}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {Object.keys(v.params).length > 0 && (
                                            <div className="flex flex-wrap gap-2 mt-2">
                                                {Object.entries(v.params).map(([k, val]) => (
                                                    <span key={k}
                                                          className="text-xs bg-gray-900 text-gray-400 px-2.5 py-1 rounded-lg font-mono">
                            {k}: {val}
                          </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}