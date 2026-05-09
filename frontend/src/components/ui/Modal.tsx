"use client";

import {useEffect} from "react";

interface Props {
    open: boolean;
    onClose: () => void;
    title: string;
    subtitle?: string;
    children: React.ReactNode;
    width?: string;
}

export function Modal({open, onClose, title, subtitle, children, width = "max-w-4xl"}: Props) {
    // Close on Escape key
    useEffect(() => {
        if (!open) return;
        const handler = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [open, onClose]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/70 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Panel */}
            <div
                className={`relative w-full ${width} bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl flex flex-col max-h-[85vh]`}>
                {/* Header */}
                <div className="flex items-start justify-between px-6 py-5 border-b border-gray-800 shrink-0">
                    <div>
                        <h2 className="text-lg font-semibold text-white">{title}</h2>
                        {subtitle && <p className="text-sm text-gray-400 mt-0.5">{subtitle}</p>}
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-white transition-colors ml-4 mt-0.5"
                        aria-label="Close"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                  d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                </div>

                {/* Scrollable content */}
                <div className="overflow-y-auto px-6 py-5">
                    {children}
                </div>
            </div>
        </div>
    );
}