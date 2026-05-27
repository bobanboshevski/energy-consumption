import type {ShapVariant} from "@/types/explainability";
import {VARIANT_CONFIG, ALL_VARIANTS} from "./constants";

interface Props {
    visible: Set<ShapVariant>;
    available: Set<ShapVariant>;   // variants that actually loaded successfully
    onChange: (v: ShapVariant) => void;
}

export function VariantSelector({visible, available, onChange}: Props) {
    return (
        <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Show variants</p>
            <div className="flex gap-3 flex-wrap">
                {ALL_VARIANTS.map((v) => {
                    const cfg = VARIANT_CONFIG[v];
                    const isAvailable = available.has(v);
                    const isVisible = visible.has(v);

                    return (
                        <button
                            key={v}
                            onClick={() => isAvailable && onChange(v)}
                            disabled={!isAvailable}
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                                !isAvailable
                                    ? "bg-gray-800/30 border-gray-700 text-gray-600 cursor-not-allowed"
                                    : isVisible
                                        ? `${cfg.badgeBg} ${cfg.badgeBorder} ${cfg.badgeText}`
                                        : "bg-gray-800/30 border-gray-700 text-gray-500 hover:border-gray-600"
                            }`}
                        >
              <span
                  className="w-2 h-2 rounded-full"
                  style={{backgroundColor: isAvailable && isVisible ? cfg.color : "#374151"}}
              />
                            {cfg.label}
                            {!isAvailable && (
                                <span className="text-gray-600 font-normal">· unavailable</span>
                            )}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}