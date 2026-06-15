"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

const LINKS = [
    { href: "/", label: "Forecast" },
    { href: "/admin", label: "Admin" },
];

export function NavBar() {
    const pathname = usePathname();

    return (
        <header className="sticky top-0 z-50 bg-gray-950">
            <div className="max-w-7xl mx-auto px-6 flex items-center justify-between h-14">

                <Link href="/" className="flex items-center gap-3 group">
                    <div className="relative">
                        <div className="absolute inset-0 rounded-xl bg-blue-500/40 blur-md
                                         opacity-0 scale-50
                                         group-hover:opacity-100 group-hover:scale-125
                                         transition-all duration-500 ease-out" />
                        <div className="relative w-8 h-8 rounded-xl flex items-center justify-center
                                        bg-gradient-to-br from-blue-500 to-blue-700
                                        group-hover:from-blue-400 group-hover:to-blue-600
                                        transition-colors duration-300">
                            <span className="text-white text-sm">⚡</span>
                        </div>
                    </div>
                    <div className="flex flex-col leading-none gap-0.5">
                        <span className="font-semibold text-sm tracking-tight text-white
                                         group-hover:text-blue-100 transition-colors duration-300">
                            Energy Demand Forecast
                        </span>
                        <span className="text-[10px] text-gray-600 tracking-widest uppercase">
                            Slovenia
                        </span>
                    </div>
                </Link>

                {/* ── Segmented control with glowing active pill ──────────── */}
                <nav
                    className="flex items-center gap-0.5 rounded-xl p-1"
                    style={{
                        background: "rgba(17, 24, 39, 0.9)",
                        boxShadow: "0 0 0 1px rgba(255,255,255,0.07)",
                    }}
                >
                    {LINKS.map(({ href, label }) => {
                        const isActive =
                            href === "/" ? pathname === "/" : pathname.startsWith(href);

                        return (
                            <Link
                                key={href}
                                href={href}
                                className="relative px-4 py-1.5 rounded-lg text-sm font-medium"
                            >
                                {isActive && (
                                    <motion.div
                                        layoutId="nav-pill"
                                        className="absolute inset-0 rounded-lg"
                                        style={{
                                            background:
                                                "linear-gradient(135deg, rgba(59,130,246,0.2), rgba(37,99,235,0.07))",
                                        }}
                                        animate={{
                                            boxShadow: [
                                                "inset 0 1px 0 rgba(147,197,253,0.16), 0 0 0 1px rgba(96,165,250,0.22), 0 4px 20px rgba(59,130,246,0.10)",
                                                "inset 0 1px 0 rgba(147,197,253,0.22), 0 0 0 1px rgba(96,165,250,0.36), 0 4px 24px rgba(59,130,246,0.20)",
                                                "inset 0 1px 0 rgba(147,197,253,0.16), 0 0 0 1px rgba(96,165,250,0.22), 0 4px 20px rgba(59,130,246,0.10)",
                                            ],
                                        }}
                                        transition={{
                                            layout: { type: "spring", stiffness: 400, damping: 32 },
                                            boxShadow: {
                                                duration: 2.5,
                                                repeat: Infinity,
                                                ease: "easeInOut",
                                            },
                                        }}
                                    />
                                )}

                                <span className={`relative z-10 transition-colors duration-200 ${
                                    isActive ? "text-blue-200" : "text-gray-500 hover:text-gray-200"
                                }`}>
                                    {label}
                                </span>
                            </Link>
                        );
                    })}
                </nav>

            </div>

            <div className="h-px bg-gradient-to-r from-transparent via-blue-900/100 to-transparent" />
        </header>
    );
}