import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Navigation } from '@/components/navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'System Threat Forecaster | Malware Prediction System',
  description: 'AI-powered malware detection system using ML and Deep Learning. Predict system threats with 7 ML models and 4 state-of-the-art DL architectures.',
  keywords: ['malware detection', 'machine learning', 'deep learning', 'cybersecurity', 'threat prediction'],
  authors: [{ name: 'Milav Dabgar' }],
  openGraph: {
    title: 'System Threat Forecaster',
    description: 'AI-powered malware detection using ML and Deep Learning',
    url: 'https://stf.milav.in',
    siteName: 'System Threat Forecaster',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Navigation />
        <main className="min-h-screen">
          {children}
        </main>
        <footer className="border-t py-6 md:py-0">
          <div className="container flex flex-col items-center justify-between gap-4 md:h-16 md:flex-row">
            <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
              Built by{' '}
              <a
                href="https://milav.in"
                target="_blank"
                rel="noreferrer"
                className="font-medium underline underline-offset-4"
              >
                Milav Dabgar
              </a>
              . AICTE QIP Deep Learning Project.
            </p>
          </div>
        </footer>
      </body>
    </html>
  )
}
