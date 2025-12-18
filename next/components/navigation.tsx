'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { Brain, Home, Zap, Info } from 'lucide-react'

export function Navigation() {
  const pathname = usePathname()

  const routes = [
    {
      href: '/',
      label: 'Home',
      icon: Home,
      active: pathname === '/',
    },
    {
      href: '/predict',
      label: 'Predict',
      icon: Zap,
      active: pathname === '/predict',
    },
    {
      href: '/models',
      label: 'Models',
      icon: Brain,
      active: pathname === '/models',
    },
    {
      href: '/about',
      label: 'About',
      icon: Info,
      active: pathname === '/about',
    },
  ]

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 flex">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <Brain className="h-6 w-6" />
            <span className="hidden font-bold sm:inline-block">
              System Threat Forecaster
            </span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            {routes.map((route) => (
              <Link
                key={route.href}
                href={route.href}
                className={cn(
                  'transition-colors hover:text-foreground/80 flex items-center gap-2',
                  route.active ? 'text-foreground' : 'text-foreground/60'
                )}
              >
                <route.icon className="h-4 w-4" />
                {route.label}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}
