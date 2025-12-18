'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Brain, Cpu, TrendingUp, Layers } from 'lucide-react'
import { useEffect, useState } from 'react'

export default function ModelsPage() {
  const [modelPerformance, setModelPerformance] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/model_performance.json')
      .then(res => res.json())
      .then(data => {
        setModelPerformance(data)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load model performance data:', err)
        setLoading(false)
      })
  }, [])

  if (loading || !modelPerformance) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading model data...</p>
        </div>
      </div>
    )
  }
  
  // Transform ML models data
  const mlModelsData = Object.entries(modelPerformance.ml_models).map(([key, data]: [string, any]) => ({
    key,
    name: key.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
    accuracy: (data.validation_accuracy * 100).toFixed(2) + '%',
    type: key === 'lightgbm' ? 'Gradient Boosting' :
          key === 'random_forest' ? 'Ensemble' :
          key === 'ada_boost' ? 'Boosting' :
          key === 'decision_tree' ? 'Tree-based' :
          key === 'logistic_regression' ? 'Linear' :
          key === 'naive_bayes' ? 'Probabilistic' :
          key === 'sgd' ? 'Linear' : 'Other',
    f1Score: data.f1_score.toFixed(4),
    precision: data.precision.toFixed(4),
    hyperparameters: data.hyperparameters
  })).sort((a: any, b: any) => parseFloat(b.accuracy) - parseFloat(a.accuracy))

  const mlModels = mlModelsData.map((model: any, index: number) => ({
    ...model,
    rank: index + 1
  }))

  // Transform DL models data
  const dlModelsData = Object.entries(modelPerformance.dl_models).map(([key, data]: [string, any]) => ({
    key,
    name: key.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
    accuracy: (data.validation_accuracy * 100).toFixed(2) + '%',
    architecture: data.architecture,
    parameters: data.total_parameters.toLocaleString(),
    trainableParams: data.trainable_parameters.toLocaleString(),
    f1Score: data.final_val_f1.toFixed(4),
    loss: data.best_val_loss.toFixed(4),
    hyperparameters: data.hyperparameters,
    training: data.training
  })).sort((a: any, b: any) => parseFloat(b.accuracy) - parseFloat(a.accuracy))

  const dlModels = dlModelsData.map((model: any, index: number) => ({
    ...model,
    rank: index + 1
  }))

  // Calculate total parameters
  const totalDLParams = dlModelsData.reduce((sum: number, model: any) => 
    sum + parseInt(model.parameters.replace(/,/g, '')), 0
  )

  const bestModel = modelPerformance.best_overall
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Model Architecture
          </h1>
          <p className="text-lg text-gray-600">
            Comprehensive analysis of {mlModels.length + dlModels.length} models ({mlModels.length} ML + {dlModels.length} DL) trained on {modelPerformance.metadata.features}-feature malware dataset
          </p>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Brain className="h-4 w-4 text-blue-600" />
                ML Models
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{mlModels.length}</div>
              <p className="text-xs text-gray-500">Traditional algorithms</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Cpu className="h-4 w-4 text-purple-600" />
                DL Models
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{dlModels.length}</div>
              <p className="text-xs text-gray-500">Neural networks</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-green-600" />
                Best Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(bestModel.accuracy * 100).toFixed(2)}%</div>
              <p className="text-xs text-gray-500">{bestModel.model.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')} ({bestModel.type})</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Layers className="h-4 w-4 text-orange-600" />
                Total Parameters
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(totalDLParams / 1000000).toFixed(1)}M</div>
              <p className="text-xs text-gray-500">DL models combined</p>
            </CardContent>
          </Card>
        </div>

        {/* Model Details Tabs */}
        <Tabs defaultValue="ml" className="space-y-4">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="ml">Machine Learning</TabsTrigger>
            <TabsTrigger value="dl">Deep Learning</TabsTrigger>
          </TabsList>
          
          <TabsContent value="ml" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Traditional Machine Learning Models</CardTitle>
                <CardDescription>
                  Ensemble and boosting algorithms optimized for tabular data
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">Rank</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>F1 Score</TableHead>
                      <TableHead>Precision</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mlModels.map((model) => (
                      <TableRow key={model.key}>
                        <TableCell className="font-medium">{model.rank}</TableCell>
                        <TableCell className="font-medium">{model.name}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{model.type}</Badge>
                        </TableCell>
                        <TableCell>
                          <span className={`font-semibold ${
                            model.rank === 1 ? 'text-green-600' : 
                            model.rank <= 3 ? 'text-blue-600' : 
                            'text-gray-600'
                          }`}>
                            {model.accuracy}
                          </span>
                        </TableCell>
                        <TableCell className="text-sm">{model.f1Score}</TableCell>
                        <TableCell className="text-sm">{model.precision}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Hyperparameters</CardTitle>
                <CardDescription>Detailed configuration for each ML model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mlModels.map((model) => (
                    <div key={model.key} className="border-b pb-4 last:border-0">
                      <h4 className="font-semibold mb-2 flex items-center gap-2">
                        {model.name}
                        {model.rank === 1 && <Badge className="bg-green-600">Best ML</Badge>}
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                        {Object.entries(model.hyperparameters).map(([key, value]) => (
                          <div key={key} className="bg-gray-50 p-2 rounded">
                            <span className="text-gray-600 block text-xs">{key}</span>
                            <span className="font-medium">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Training Samples</p>
                    <p className="text-lg font-semibold">{modelPerformance.metadata.training_samples.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Validation Samples</p>
                    <p className="text-lg font-semibold">{modelPerformance.metadata.validation_samples.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Features</p>
                    <p className="text-lg font-semibold">{modelPerformance.metadata.features} dimensions</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Preprocessing</p>
                    <p className="text-lg font-semibold">{modelPerformance.preprocessing.scaling}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="dl" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Deep Learning Neural Networks</CardTitle>
                <CardDescription>
                  Modern architectures with attention mechanisms and residual connections
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">Rank</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead>Architecture</TableHead>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>F1 Score</TableHead>
                      <TableHead>Parameters</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dlModels.map((model) => (
                      <TableRow key={model.key}>
                        <TableCell className="font-medium">{model.rank}</TableCell>
                        <TableCell className="font-medium">{model.name}</TableCell>
                        <TableCell>
                          <Badge variant="outline">{model.architecture}</Badge>
                        </TableCell>
                        <TableCell>
                          <span className={`font-semibold ${
                            model.rank === 1 ? 'text-green-600' : 
                            model.rank <= 2 ? 'text-blue-600' : 
                            'text-gray-600'
                          }`}>
                            {model.accuracy}
                          </span>
                        </TableCell>
                        <TableCell className="text-sm">{model.f1Score}</TableCell>
                        <TableCell className="text-sm font-mono">{model.parameters}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Hyperparameters & Architecture</CardTitle>
                <CardDescription>Detailed configuration for each DL model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dlModels.map((model) => (
                    <div key={model.key} className="border-b pb-4 last:border-0">
                      <h4 className="font-semibold mb-2 flex items-center gap-2">
                        {model.name}
                        {model.rank === 1 && <Badge className="bg-green-600">Best DL</Badge>}
                        <Badge variant="outline" className="ml-auto">{model.parameters} params</Badge>
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm mb-2">
                        {Object.entries(model.hyperparameters).map(([key, value]) => (
                          <div key={key} className="bg-gray-50 p-2 rounded">
                            <span className="text-gray-600 block text-xs">{key}</span>
                            <span className="font-medium">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                          </div>
                        ))}
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-sm">
                        <div className="bg-blue-50 p-2 rounded">
                          <span className="text-gray-600 block text-xs">Device</span>
                          <span className="font-medium">{model.training.device}</span>
                        </div>
                        <div className="bg-blue-50 p-2 rounded">
                          <span className="text-gray-600 block text-xs">Scheduler</span>
                          <span className="font-medium">{model.training.scheduler}</span>
                        </div>
                        <div className="bg-blue-50 p-2 rounded">
                          <span className="text-gray-600 block text-xs">Early Stopping</span>
                          <span className="font-medium">{model.training.early_stopping_patience} epochs</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Epochs</p>
                    <p className="text-lg font-semibold">{dlModels[0]?.hyperparameters.epochs} epochs</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Batch Size</p>
                    <p className="text-lg font-semibold">{dlModels[0]?.hyperparameters.batch_size}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Optimizer</p>
                    <p className="text-lg font-semibold">{dlModels[0]?.hyperparameters.optimizer.toUpperCase()}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Learning Rate</p>
                    <p className="text-lg font-semibold">{dlModels[0]?.hyperparameters.learning_rate}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Hardware</p>
                    <p className="text-lg font-semibold">{dlModels[0]?.training.device.toUpperCase()}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Dropout Rate</p>
                    <p className="text-lg font-semibold">{dlModels[0]?.hyperparameters.dropout_rate}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Performance Analysis */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Performance Analysis</CardTitle>
            <CardDescription>Key insights from model comparison</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">🏆 Top Performers</h3>
              <p className="text-sm text-gray-600">
                {mlModels[0].name} achieves the highest accuracy ({mlModels[0].accuracy}) among all models, 
                followed by {mlModels[1].name} ({mlModels[1].accuracy}). Gradient boosting algorithms dominate 
                the top positions due to their ability to handle complex feature interactions in tabular data.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">🧠 Deep Learning Performance</h3>
              <p className="text-sm text-gray-600">
                {dlModels[0].name} ({dlModels[0].accuracy}) leads the neural network models with {dlModels[0].parameters} parameters, 
                demonstrating that deep learning can effectively learn patterns in tabular malware data. The attention-based 
                architectures show competitive performance while maintaining interpretability.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">⚡ Model Efficiency</h3>
              <p className="text-sm text-gray-600">
                Traditional ML models train in seconds while achieving top accuracy, making them ideal for rapid iteration. 
                Deep learning models require ~15-30 minutes per model on Apple Silicon MPS but offer potential for 
                improvement with architectural innovations and larger datasets.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">🔄 Dataset Insights</h3>
              <p className="text-sm text-gray-600">
                Training on {modelPerformance.metadata.training_samples.toLocaleString()} samples with {modelPerformance.metadata.features} features, 
                models achieve ~{mlModels[0].accuracy} accuracy. The {modelPerformance.preprocessing.numeric_features} numeric and {modelPerformance.preprocessing.categorical_features} categorical 
                features capture device characteristics, security settings, and behavioral patterns for malware detection.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
