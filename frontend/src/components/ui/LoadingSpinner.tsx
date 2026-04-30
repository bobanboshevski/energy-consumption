export function LoadingSpinner({ text = "Loading..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      <p className="text-gray-400 text-sm">{text}</p>
    </div>
  );
}

export function InlineSpinner() {
  return <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin inline-block" />;
}