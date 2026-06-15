import type {Metadata} from "next";
import {Inter} from "next/font/google";
import "./globals.css";
import {NavBar} from "@/components/ui/NavBar";

const inter = Inter({subsets: ["latin"]});

export const metadata: Metadata = {
    title: "Energy Demand Forecast — Slovenia",
    description: "AI-powered energy demand prediction system",
};

export default function RootLayout({children}: { children: React.ReactNode }) {
    return (
        <html lang="en" className="dark">
        <body className={`${inter.className} bg-gray-950 text-gray-100 min-h-screen`}>
        <NavBar/>
        <main>{children}</main>
        </body>
        </html>
    );
}