/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  poweredByHeader: false,
  reactStrictMode: true,
  env: {
    SITE_URL: process.env.SITE_URL || 'http://localhost:3000',
  },
}

module.exports = nextConfig
