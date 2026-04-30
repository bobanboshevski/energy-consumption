interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  trend?: "up" | "down" | "neutral";
  accent?: string;
}

export function StatCard({ label, value, sub, trend, accent = "text-blue-400" }: StatCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-2">{label}</p>
      <p className={`text-3xl font-bold font-mono ${accent}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1.5">{sub}</p>}
    </div>
  );
}