"use client";
import Link from "next/link";
import {usePathname} from "next/navigation";
import {motion} from "framer-motion";

const LINKS = [
    {href: "/", label: "Forecast"},
    {href: "/admin", label: "Admin"},
];

export function NavBar() {
    const pathname = usePathname();

    return (
        <header className="sticky top-0 z-50 bg-gray-950">
            <div className="max-w-7xl mx-auto px-6 flex items-center justify-between h-14">

                {/* ── Logo ─────────────────────────────────────────────────
                    Glow is via box-shadow, NOT filter/blur.
                    filter: blur() on any descendant of sticky breaks Safari. */}
                <Link href="/" className="flex items-center gap-3 group">
                    <div
                        className="w-8 h-8 rounded-xl flex items-center justify-center
                                   bg-gradient-to-br from-blue-500 to-blue-700
                                   group-hover:from-blue-400 group-hover:to-blue-600
                                   transition-all duration-300"
                        style={{
                            boxShadow: "0 0 0 0 rgba(59,130,246,0)",
                        }}
                        onMouseEnter={(e) => {
                            (e.currentTarget as HTMLElement).style.boxShadow =
                                "0 0 18px 4px rgba(59,130,246,0.35)";
                        }}
                        onMouseLeave={(e) => {
                            (e.currentTarget as HTMLElement).style.boxShadow =
                                "0 0 0 0 rgba(59,130,246,0)";
                        }}
                    >
                        <span className="text-white text-sm">⚡</span>
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

                {/* ── Segmented control ────────────────────────────────────── */}
                <nav
                    className="flex items-center gap-0.5 rounded-xl p-1"
                    style={{
                        background: "rgba(17, 24, 39, 0.9)",
                        boxShadow: "0 0 0 1px rgba(255,255,255,0.07)",
                    }}
                >
                    {LINKS.map(({href, label}) => {
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
                                            boxShadow:
                                                "inset 0 1px 0 rgba(147,197,253,0.16), 0 0 0 1px rgba(96,165,250,0.25), 0 4px 20px rgba(59,130,246,0.12)",
                                        }}
                                        transition={{
                                            type: "spring",
                                            stiffness: 400,
                                            damping: 32,
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
            <div className="h-px bg-gradient-to-r from-transparent via-blue-900/60 to-transparent"/>
        </header>
    );
}