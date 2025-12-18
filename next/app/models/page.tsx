'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Brain, Cpu, TrendingUp, Layers } from 'lucide-react'

const mlModels = [
  { name: 'LightGBM', accuracy: '63.0%', type: 'Gradient Boosting', features: 'Handles categorical data, fast training', rank: 1 },
  { name: 'XGBoost', accuracy: '62.8%', type: 'Gradient Boosting', features: 'Regularization, parallel processing', rank: 2 },
  { name: 'Random Forest', accuracy: '62.5%', type: 'Ensemble', features: 'Robust to overfitting, feature importance', rank: 3 },
  { name: 'Extra Trees', accuracy: '62.3%', type: 'Ensemble', features: 'More randomization, faster training', rank: 4 },
  { name: 'CatBoost', accuracy: '62.1%', type: 'Gradient Boosting', features: 'Categorical feature handling, GPU support', rank: 5 },
  { name: 'AdaBoost', accuracy: '59.5%', type: 'Boosting', features: 'Adaptive boosting, combines weak learners', rank: 6 },
  { name: 'Gradient Boosting', accuracy: '58.2%', type: 'Boosting', features: 'Sequential model building, loss optimization', rank: 7 },
]

const dlModels = [
  { 
    name: 'Attention Network', 
    accuracy: '61.73%', 
    architecture: 'Transformer-based', 
    features: 'Self-attention mechanism, contextual feature learning',
    parameters: '~2.5M',
    layers: 'Multi-head attention, feed-forward networks',
    rank: 1
  },
  { 
    name: 'Deep MLP', 
    accuracy: '61.72%', 
    architecture: 'Feed-forward', 
    features: 'Multiple hidden layers, dropout regularization',
    parameters: '~2.2M',
    layers: '5 hidden layers (512, 256, 128, 64, 32)',
    rank: 2
  },
  { 
    name: 'Residual Network', 
    accuracy: '61.48%', 
    architecture: 'ResNet-style', 
    features: 'Skip connections, gradient flow optimization',
    parameters: '~2.8M',
    layers: 'Residual blocks with batch normalization',
    rank: 3
  },
  { 
    name: 'FT-Transformer', 
    accuracy: '~61.0%', 
    architecture: 'Feature Tokenizer + Transformer', 
    features: 'Tabular data transformer, feature embedding',
    parameters: '~3.1M',
    layers: 'Feature tokenization + multi-layer transformer',
    rank: 4
  },
]

export default function ModelsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Model Architecture
          </h1>
          <p className="text-lg text-gray-600">
            Comprehensive analysis of 11 models (7 ML + 4 DL) trained on 76-feature malware dataset
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
              <div className="text-2xl font-bold">7</div>
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
              <div className="text-2xl font-bold">4</div>
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
              <div className="text-2xl font-bold">63.0%</div>
              <p className="text-xs text-gray-500">LightGBM</p>
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
              <div className="text-2xl font-bold">~10.6M</div>
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
                      <TableHead>Key Features</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {mlModels.map((model) => (
                      <TableRow key={model.name}>
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
                        <TableCell className="text-sm text-gray-600">{model.features}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Dataset Size</p>
                    <p className="text-lg font-semibold">2,000 samples</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Features</p>
                    <p className="text-lg font-semibold">76 dimensions</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Cross-validation</p>
                    <p className="text-lg font-semibold">5-fold CV</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Preprocessing</p>
                    <p className="text-lg font-semibold">StandardScaler</p>
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
                      <TableHead>Parameters</TableHead>
                      <TableHead>Key Components</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dlModels.map((model) => (
                      <TableRow key={model.name}>
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
                        <TableCell className="text-sm font-mono">{model.parameters}</TableCell>
                        <TableCell className="text-sm text-gray-600">{model.layers}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
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
                    <p className="text-lg font-semibold">50 epochs</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Batch Size</p>
                    <p className="text-lg font-semibold">64</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Optimizer</p>
                    <p className="text-lg font-semibold">Adam</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Loss Function</p>
                    <p className="text-lg font-semibold">Cross Entropy</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Hardware</p>
                    <p className="text-lg font-semibold">Apple M3 MPS</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Training Time</p>
                    <p className="text-lg font-semibold">~30 min/model</p>
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
                LightGBM achieves the highest accuracy (63.0%) among all models, followed closely by XGBoost (62.8%). 
                Gradient boosting algorithms dominate the top positions due to their ability to handle complex feature interactions.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">🧠 Deep Learning Performance</h3>
              <p className="text-sm text-gray-600">
                Attention Network (61.73%) and Deep MLP (61.72%) show competitive performance, demonstrating that neural networks 
                can effectively learn patterns in tabular data when properly architected.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">⚡ Ensemble Methods Advantage</h3>
              <p className="text-sm text-gray-600">
                Ensemble methods (Random Forest, Extra Trees) provide robust predictions with lower variance, making them reliable 
                choices for production deployment despite slightly lower accuracy than boosting methods.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">🔄 Model Diversity</h3>
              <p className="text-sm text-gray-600">
                Combining predictions from multiple models with different learning paradigms (boosting, bagging, neural networks) 
                can potentially improve overall system accuracy through ensemble voting.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
