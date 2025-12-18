'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Loader2, RefreshCcw, Zap, AlertCircle } from 'lucide-react'
import { FeatureDisplay } from '@/components/feature-display'
import { ModelComparison } from '@/components/model-comparison'

export default function PredictPage() {
  const [sample, setSample] = useState<any>(null)
  const [predictions, setPredictions] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')

  const loadRandomSample = async () => {
    setLoading(true)
    setError('')
    setPredictions(null)
    
    try {
      const response = await fetch('/api/load-sample')
      if (!response.ok) throw new Error('Failed to load sample')
      const data = await response.json()
      setSample(data)
    } catch (err) {
      setError('Failed to load sample. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const predictAll = async () => {
    if (!sample) return
    
    setPredicting(true)
    setError('')
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sample),
      })
      
      if (!response.ok) throw new Error('Prediction failed')
      const data = await response.json()
      setPredictions(data)
    } catch (err) {
      setError('Prediction failed. Please try again.')
      console.error(err)
    } finally {
      setPredicting(false)
    }
  }

  return (
    <div className="container py-10">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Live Prediction</h1>
          <p className="text-muted-foreground">
            Load a random sample and get predictions from all 11 models
          </p>
        </div>

        {/* Load Sample */}
        <Card>
          <CardHeader>
            <CardTitle>Step 1: Load Sample Data</CardTitle>
            <CardDescription>
              Select a random sample from the dataset (100,000 samples available)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              onClick={loadRandomSample} 
              disabled={loading}
              className="gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <RefreshCcw className="w-4 h-4" />
                  Load Random Sample
                </>
              )}
            </Button>
            
            {sample && (
              <div className="mt-4 flex items-center gap-2">
                <Badge variant="outline">Sample loaded</Badge>
                <Badge variant="secondary">76 features</Badge>
                <Badge variant="secondary">
                  Actual: {sample.target === 1 ? 'Malware' : 'No Malware'}
                </Badge>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Feature Display */}
        {sample && (
          <Card>
            <CardHeader>
              <CardTitle>Sample Features</CardTitle>
              <CardDescription>
                System properties used for prediction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="grouped" className="w-full">
                <TabsList>
                  <TabsTrigger value="grouped">Grouped</TabsTrigger>
                  <TabsTrigger value="all">All Features</TabsTrigger>
                </TabsList>
                <TabsContent value="grouped">
                  <FeatureDisplay features={sample.features} mode="grouped" />
                </TabsContent>
                <TabsContent value="all">
                  <FeatureDisplay features={sample.features} mode="all" />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        )}

        {/* Predict Button */}
        {sample && !predictions && (
          <Card>
            <CardHeader>
              <CardTitle>Step 2: Run Prediction</CardTitle>
              <CardDescription>
                Get predictions from all ML and DL models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={predictAll} 
                disabled={predicting}
                size="lg"
                className="gap-2"
              >
                {predicting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Predicting with 11 models...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Predict with All Models
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {predictions && (
          <Card>
            <CardHeader>
              <CardTitle>Prediction Results</CardTitle>
              <CardDescription>
                Comparison of all model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelComparison 
                predictions={predictions} 
                actualLabel={sample.target}
              />
            </CardContent>
          </Card>
        )}

        {/* Error Display */}
        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-4 h-4" />
                <p>{error}</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
