import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Energy Demand Forecast — Slovenia",
  description: "AI-powered energy demand prediction system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-950 text-gray-100 min-h-screen`}>
        <nav className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-xs font-bold">⚡</span>
              </div>
              <span className="font-semibold text-white">EnergyForecast</span>
              <span className="text-xs text-gray-500 hidden sm:block">Slovenia</span>
            </Link>
            <div className="flex items-center gap-6 text-sm">
              <Link href="/" className="text-gray-400 hover:text-white transition-colors">Forecast</Link>
              <Link href="/admin" className="text-gray-400 hover:text-white transition-colors">Admin</Link>
            </div>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}