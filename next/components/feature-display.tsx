interface FeatureDisplayProps {
  features: Record<string, any>
  mode: 'grouped' | 'all'
}

export function FeatureDisplay({ features, mode }: FeatureDisplayProps) {
  if (mode === 'all') {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-96 overflow-y-auto">
        {Object.entries(features).map(([key, value]) => (
          <div key={key} className="border rounded-lg p-3 space-y-1">
            <div className="text-xs font-medium text-muted-foreground truncate">
              {key}
            </div>
            <div className="text-sm font-mono">{String(value)}</div>
          </div>
        ))}
      </div>
    )
  }

  // Grouped mode - categorize features
  const numerical = Object.entries(features).filter(([_, v]) => typeof v === 'number')
  const categorical = Object.entries(features).filter(([_, v]) => typeof v !== 'number')

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-3">Numerical Features ({numerical.length})</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-64 overflow-y-auto">
          {numerical.map(([key, value]) => (
            <div key={key} className="border rounded-lg p-3 space-y-1">
              <div className="text-xs font-medium text-muted-foreground truncate">
                {key}
              </div>
              <div className="text-sm font-mono">{String(value)}</div>
            </div>
          ))}
        </div>
      </div>
      
      <div>
        <h3 className="text-lg font-semibold mb-3">Categorical Features ({categorical.length})</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-64 overflow-y-auto">
          {categorical.map(([key, value]) => (
            <div key={key} className="border rounded-lg p-3 space-y-1">
              <div className="text-xs font-medium text-muted-foreground truncate">
                {key}
              </div>
              <div className="text-sm font-mono">{String(value)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
