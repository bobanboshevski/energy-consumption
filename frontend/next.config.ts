import type {NextConfig} from "next";

const nextConfig: NextConfig = {
    output: "standalone", // optimization for docker
    allowedDevOrigins: ['172.20.10.8', 'harsh-election-esteemed.ngrok-free.dev'],
};

export default nextConfig;
