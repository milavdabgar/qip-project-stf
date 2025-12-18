import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { CheckCircle2, XCircle } from 'lucide-react'

interface ModelComparisonProps {
  predictions: {
    ml_models: Record<string, { prediction: number; confidence: number }>
    dl_models: Record<string, { prediction: number; confidence: number }>
  }
  actualLabel: number
}

export function ModelComparison({ predictions, actualLabel }: ModelComparisonProps) {
  const allModels = [
    ...Object.entries(predictions.ml_models).map(([name, data]) => ({
      name,
      type: 'ML',
      ...data,
    })),
    ...Object.entries(predictions.dl_models).map(([name, data]) => ({
      name,
      type: 'DL',
      ...data,
    })),
  ]

  // Sort by confidence
  const sortedModels = allModels.sort((a, b) => b.confidence - a.confidence)

  const formatModelName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  const isCorrect = (prediction: number) => prediction === actualLabel

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex gap-4 flex-wrap">
        <Badge variant="outline" className="text-sm">
          Actual: {actualLabel === 1 ? 'Malware' : 'No Malware'}
        </Badge>
        <Badge variant="secondary" className="text-sm">
          Correct: {sortedModels.filter(m => isCorrect(m.prediction)).length} / {sortedModels.length}
        </Badge>
      </div>

      {/* Table */}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Prediction</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead className="text-right">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedModels.map((model, idx) => (
              <TableRow key={idx} className={isCorrect(model.prediction) ? 'bg-green-50 dark:bg-green-950/20' : ''}>
                <TableCell className="font-medium">
                  {formatModelName(model.name)}
                  {model.name === 'lightgbm' && (
                    <Badge className="ml-2 bg-yellow-500 text-xs">Best</Badge>
                  )}
                </TableCell>
                <TableCell>
                  <Badge variant={model.type === 'ML' ? 'default' : 'secondary'}>
                    {model.type}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge variant={model.prediction === 1 ? 'destructive' : 'default'}>
                    {model.prediction === 1 ? 'Malware' : 'No Malware'}
                  </Badge>
                </TableCell>
                <TableCell>
                  <span className="font-mono text-sm">
                    {(model.confidence * 100).toFixed(1)}%
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  {isCorrect(model.prediction) ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500 inline" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-500 inline" />
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
