import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { parse } from 'csv-parse/sync'

export async function GET() {
  try {
    // Path to train.csv
    const csvPath = path.join(process.cwd(), '..', 'kaggle', 'input', 'System-Threat-Forecaster', 'train.csv')
    
    // Read CSV file
    const fileContent = fs.readFileSync(csvPath, 'utf-8')
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
    })

    // Get random sample
    const randomIndex = Math.floor(Math.random() * records.length)
    const sample = records[randomIndex]

    // Separate target from features (CSV has 'target' column, not 'HasDetections')
    const { target, ...features } = sample

    // Convert numeric strings to numbers
    const processedFeatures: Record<string, any> = {}
    Object.entries(features).forEach(([key, value]) => {
      const numValue = Number(value)
      processedFeatures[key] = isNaN(numValue) ? value : numValue
    })

    return NextResponse.json({
      features: processedFeatures,
      target: Number(target),
      sampleIndex: randomIndex,
    })
  } catch (error) {
    console.error('Error loading sample:', error)
    return NextResponse.json(
      { error: 'Failed to load sample' },
      { status: 500 }
    )
  }
}
