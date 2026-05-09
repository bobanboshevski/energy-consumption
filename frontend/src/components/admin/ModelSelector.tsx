import type {ModelView} from "@/hooks/useAdmin";

interface Props {
    selected: ModelView;
    onChange: (view: ModelView) => void;
}

const OPTIONS: { id: ModelView; label: string; description: string }[] = [
    {
        id: "multivariate",
        label: "Multivariate",
        description: "Weather + energy demand · 16-day forecast",
    },
    {
        id: "univariate",
        label: "Univariate",
        description: "Energy demand only · 365-day forecast",
    },
];

export function ModelSelector({selected, onChange}: Props) {
    return (
        <div className="flex gap-2 mb-6">
            {OPTIONS.map((o) => (
                <button
                    key={o.id}
                    onClick={() => onChange(o.id)}
                    className={`flex-1 text-left px-4 py-3 rounded-xl border transition-all ${
                        selected === o.id
                            ? "bg-blue-950/40 border-blue-700"
                            : "bg-gray-800/40 border-gray-700 hover:border-gray-600"
                    }`}
                >
                    <p className={`text-sm font-semibold ${selected === o.id ? "text-blue-400" : "text-gray-300"}`}>
                        {o.label}
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">{o.description}</p>
                </button>
            ))}
        </div>
    );
}