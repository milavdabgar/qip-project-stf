'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { GraduationCap, Award, Code, Database, Brain, Server } from 'lucide-react'

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            About System Threat Forecaster
          </h1>
          <p className="text-lg text-gray-600">
            An AI-powered malware detection system developed as part of AICTE Quality Improvement Programme
          </p>
        </div>

        {/* Project Overview */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GraduationCap className="h-5 w-5 text-blue-600" />
              Project Overview
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Mission</h3>
              <p className="text-gray-600">
                To develop an advanced malware detection system leveraging both traditional machine learning and 
                modern deep learning techniques to identify and classify potential security threats in real-time.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Approach</h3>
              <p className="text-gray-600">
                We implemented and compared 11 different models (7 ML + 4 DL) to find the optimal balance between 
                accuracy, speed, and resource utilization. Each model brings unique strengths to the ensemble, 
                creating a robust multi-model prediction system.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Dataset</h3>
              <p className="text-gray-600">
                Our models are trained on a comprehensive malware dataset containing 2,000 samples with 76 features 
                extracted from system behavior analysis, including OS configuration, installation patterns, and 
                update mechanisms.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* AICTE QIP Program */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-5 w-5 text-purple-600" />
              AICTE Quality Improvement Programme
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-600">
              This project is developed under the <strong>All India Council for Technical Education (AICTE) 
              Quality Improvement Programme (QIP)</strong>, which aims to enhance the quality of technical 
              education in India through research and innovation.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">Research Focus</h4>
                <p className="text-sm text-blue-700">
                  Deep Learning for Cybersecurity
                </p>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <h4 className="font-semibold text-purple-900 mb-2">Duration</h4>
                <p className="text-sm text-purple-700">
                  2024-2025 Academic Year
                </p>
              </div>
              <div className="p-4 bg-indigo-50 rounded-lg">
                <h4 className="font-semibold text-indigo-900 mb-2">Institution</h4>
                <p className="text-sm text-indigo-700">
                  Computer Science Department
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Technology Stack */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Code className="h-5 w-5 text-green-600" />
              Technology Stack
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Brain className="h-4 w-4 text-blue-600" />
                  Machine Learning
                </h3>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge>Python 3.14</Badge>
                    <span className="text-sm text-gray-600">Core language</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge>scikit-learn 1.8</Badge>
                    <span className="text-sm text-gray-600">ML algorithms</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge>PyTorch 2.9</Badge>
                    <span className="text-sm text-gray-600">Deep learning</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge>LightGBM</Badge>
                    <span className="text-sm text-gray-600">Gradient boosting</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge>XGBoost</Badge>
                    <span className="text-sm text-gray-600">Extreme boosting</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Server className="h-4 w-4 text-purple-600" />
                  Web Application
                </h3>
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Next.js 14</Badge>
                    <span className="text-sm text-gray-600">React framework</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">TypeScript</Badge>
                    <span className="text-sm text-gray-600">Type safety</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Tailwind CSS</Badge>
                    <span className="text-sm text-gray-600">Styling</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">shadcn/ui</Badge>
                    <span className="text-sm text-gray-600">UI components</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">Lucide Icons</Badge>
                    <span className="text-sm text-gray-600">Icon library</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <Database className="h-4 w-4 text-orange-600" />
                Data Processing
              </h3>
              <div className="flex flex-wrap gap-2">
                <Badge>pandas</Badge>
                <Badge>NumPy</Badge>
                <Badge>joblib</Badge>
                <Badge>csv-parse</Badge>
                <Badge>StandardScaler</Badge>
                <Badge>LabelEncoder</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Key Features */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Key Features</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">🎯 Multi-Model Ensemble</h3>
                <p className="text-sm text-gray-600">
                  Combines predictions from 11 different models for robust threat detection
                </p>
              </div>
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">⚡ Real-Time Prediction</h3>
                <p className="text-sm text-gray-600">
                  Instant classification of system profiles through optimized inference pipeline
                </p>
              </div>
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">📊 Comprehensive Analysis</h3>
                <p className="text-sm text-gray-600">
                  76-dimensional feature analysis covering OS, architecture, and update patterns
                </p>
              </div>
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">🔒 Production Ready</h3>
                <p className="text-sm text-gray-600">
                  Fully deployed system with API integration and web interface
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Developer */}
        <Card>
          <CardHeader>
            <CardTitle>Developer</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-start gap-4">
              <div className="h-16 w-16 rounded-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white text-2xl font-bold">
                MD
              </div>
              <div>
                <h3 className="font-semibold text-lg">Milav Dabgar</h3>
                <p className="text-gray-600 mb-2">AICTE QIP Deep Learning Project</p>
                <p className="text-sm text-gray-500">
                  Research focus: Applying deep learning techniques to cybersecurity challenges, 
                  with emphasis on malware detection and threat classification.
                </p>
                <div className="mt-3 flex gap-2">
                  <Badge>Machine Learning</Badge>
                  <Badge>Deep Learning</Badge>
                  <Badge>Cybersecurity</Badge>
                  <Badge>Full Stack Development</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer Note */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>System Threat Forecaster © 2025 | Deployed at stf.milav.in</p>
          <p className="mt-1">Built with ❤️ for AICTE QIP Programme</p>
        </div>
      </div>
    </div>
  )
}
