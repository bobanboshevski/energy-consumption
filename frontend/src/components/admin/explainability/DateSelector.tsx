import type {ShapExplanation} from "@/types/explainability";

interface Props {
    explanations: ShapExplanation[];
    selected: string;
    onChange: (date: string) => void;
}

function formatDate(d: string) {
    return new Date(d).toLocaleDateString("en-GB", {
        weekday: "short", day: "numeric", month: "short",
    });
}

export function DateSelector({explanations, selected, onChange}: Props) {
    return (
        <div>
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Forecast date</p>
            <div className="flex flex-wrap gap-2">
                {explanations.map((e) => (
                    <button
                        key={e.date}
                        onClick={() => onChange(e.date)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                            selected === e.date
                                ? "bg-blue-600 text-white border-blue-500"
                                : "bg-gray-800/60 text-gray-400 border-gray-700 hover:border-gray-500 hover:text-gray-200"
                        }`}
                    >
                        {formatDate(e.date)}
                    </button>
                ))}
            </div>
        </div>
    );
}