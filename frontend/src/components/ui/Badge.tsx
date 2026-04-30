import { DemandCategory } from "@/types";

const config: Record<DemandCategory, { bg: string; text: string; dot: string; label: string }> = {
  low:    { bg: "bg-emerald-950", text: "text-emerald-400", dot: "bg-emerald-400", label: "Low" },
  medium: { bg: "bg-amber-950",   text: "text-amber-400",   dot: "bg-amber-400",   label: "Medium" },
  high:   { bg: "bg-red-950",     text: "text-red-400",     dot: "bg-red-400",     label: "High" },
};

export function DemandBadge({ category }: { category: DemandCategory }) {
  const c = config[category];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const isFinished = status === "FINISHED";
  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
      isFinished ? "bg-emerald-950 text-emerald-400" : "bg-amber-950 text-amber-400"
    }`}>
      {status}
    </span>
  );
}