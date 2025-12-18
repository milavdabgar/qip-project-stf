'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Brain, Cpu, Shield, TrendingUp, Zap, Database } from 'lucide-react'

export default function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-primary/10 via-primary/5 to-background py-20 md:py-32">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center space-y-8 text-center">
            <div className="space-y-4 max-w-3xl">
              <Badge className="mb-4" variant="outline">
                <Shield className="w-3 h-3 mr-1" />
                AICTE QIP Deep Learning Project
              </Badge>
              <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl bg-clip-text text-transparent bg-gradient-to-r from-primary to-blue-600">
                System Threat Forecaster
              </h1>
              <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
                AI-powered malware detection using 7 Machine Learning models and 4 cutting-edge Deep Learning architectures. 
                Predict system threats with 63% accuracy using LightGBM.
              </p>
            </div>
            <div className="flex flex-col gap-4 sm:flex-row">
              <Link href="/predict">
                <Button size="lg" className="gap-2">
                  <Zap className="w-4 h-4" />
                  Try Live Prediction
                </Button>
              </Link>
              <Link href="/models">
                <Button size="lg" variant="outline" className="gap-2">
                  <Brain className="w-4 h-4" />
                  Explore Models
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 bg-muted/50">
        <div className="container px-4 md:px-6">
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">ML Models</CardTitle>
                <Brain className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">7</div>
                <p className="text-xs text-muted-foreground">
                  Traditional algorithms
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">DL Models</CardTitle>
                <Cpu className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">4</div>
                <p className="text-xs text-muted-foreground">
                  Neural architectures
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Best Accuracy</CardTitle>
                <TrendingUp className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">63.0%</div>
                <p className="text-xs text-muted-foreground">
                  LightGBM performance
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-sm font-medium">Dataset</CardTitle>
                <Database className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">100K</div>
                <p className="text-xs text-muted-foreground">
                  Samples, 76 features
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container px-4 md:px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
              Model Performance
            </h2>
            <p className="mt-4 text-muted-foreground md:text-lg max-w-2xl mx-auto">
              Comprehensive comparison of traditional ML and state-of-the-art DL approaches
            </p>
          </div>
          <div className="grid gap-8 md:grid-cols-2">
            <Card className="border-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Machine Learning Models</CardTitle>
                  <Badge variant="default">Best: 63.0%</Badge>
                </div>
                <CardDescription>Traditional algorithms excel at tabular data</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">LightGBM</span>
                  <Badge className="bg-green-500">63.0%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Random Forest</span>
                  <Badge variant="secondary">62.5%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">AdaBoost</span>
                  <Badge variant="secondary">61.8%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Decision Tree</span>
                  <Badge variant="secondary">59.2%</Badge>
                </div>
                <div className="flex items-center justify-between text-muted-foreground text-sm">
                  <span>+ 3 more models</span>
                  <Link href="/models" className="text-primary hover:underline text-xs">
                    View all →
                  </Link>
                </div>
              </CardContent>
            </Card>

            <Card className="border-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Deep Learning Models</CardTitle>
                  <Badge variant="default">Best: 61.73%</Badge>
                </div>
                <CardDescription>Neural networks with GPU acceleration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Attention Net</span>
                  <Badge className="bg-blue-500">61.73%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Deep MLP</span>
                  <Badge variant="secondary">61.72%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">FT-Transformer</span>
                  <Badge variant="secondary">61.59%</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Residual Net</span>
                  <Badge variant="secondary">61.48%</Badge>
                </div>
                <div className="text-muted-foreground text-xs mt-2">
                  <Badge variant="outline" className="mr-2">PyTorch 2.9.1</Badge>
                  <Badge variant="outline">Apple Silicon GPU</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-t from-primary/10 to-background">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                Ready to test the system?
              </h2>
              <p className="mx-auto max-w-[600px] text-muted-foreground md:text-lg">
                Load a random sample and see predictions from all 13 models in real-time
              </p>
            </div>
            <Link href="/predict">
              <Button size="lg" className="gap-2">
                <Zap className="w-4 h-4" />
                Start Predicting
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
