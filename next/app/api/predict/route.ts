import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: Request) {
  try {
    const { features } = await request.json()

    // Path to Python script and venv
    const pythonScript = path.join(process.cwd(), '..', 'predict.py')
    const venvPython = path.join(process.cwd(), '..', 'venv', 'bin', 'python')
    
    // Spawn Python process using venv
    const python = spawn(venvPython, [pythonScript])
    
    let output = ''
    let errorOutput = ''
    
    // Collect stdout
    python.stdout.on('data', (data) => {
      output += data.toString()
    })
    
    // Collect stderr
    python.stderr.on('data', (data) => {
      errorOutput += data.toString()
    })
    
    // Send features to Python script via stdin
    python.stdin.write(JSON.stringify({ features }))
    python.stdin.end()
    
    // Wait for process to complete
    await new Promise((resolve, reject) => {
      python.on('close', (code) => {
        if (code !== 0) {
          console.error('Python error:', errorOutput)
          reject(new Error(`Python process exited with code ${code}`))
        } else {
          resolve(null)
        }
      })
      
      python.on('error', (err) => {
        reject(err)
      })
    })
    
    // Parse output
    const predictions = JSON.parse(output)
    return NextResponse.json(predictions)
    
  } catch (error) {
    console.error('Error getting predictions:', error)
    return NextResponse.json(
      { error: 'Failed to get predictions' },
      { status: 500 }
    )
  }
}
