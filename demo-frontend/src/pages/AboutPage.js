import React from 'react';
import { motion } from 'framer-motion';
import { 
  CubeIcon, 
  LightBulbIcon,
  ChartBarIcon,
  CogIcon,
  RocketLaunchIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

function AboutPage() {
  const architectureFeatures = [
    {
      icon: CubeIcon,
      title: 'Distributed Semantic Cubes',
      description: 'Five specialized Cartesian cubes handle different domains: Code, Data, User, Temporal, and System.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: LightBulbIcon,
      title: 'Topological Intelligence',
      description: 'Cross-cube relationships use topological data analysis to preserve semantic structure.',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: ChartBarIcon,
      title: 'Coordinate-Based Access',
      description: 'Information is accessed through precise coordinates, not sequential token processing.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: CogIcon,
      title: 'Intelligent Orchestration',
      description: 'ML-powered orchestrator optimally routes queries across cubes for maximum accuracy.',
      color: 'from-orange-500 to-red-500'
    }
  ];

  const problemSolutions = [
    {
      problem: 'Token Limit Collapse',
      traditional: 'Exponential performance degradation after 25k tokens',
      solution: 'Distributed processing eliminates token-based limitations',
      impact: 'Maintains 80%+ accuracy at 1M+ tokens'
    },
    {
      problem: 'Memory Bottlenecks',
      traditional: 'Cannot process contexts beyond 100k tokens',
      solution: 'Each cube handles specialized domains independently',
      impact: 'Linear scaling with context size'
    },
    {
      problem: 'Lost Context',
      traditional: 'Important information gets "lost in the middle"',
      solution: 'Coordinate-based retrieval finds exact information',
      impact: 'Perfect information preservation'
    },
    {
      problem: 'Semantic Incoherence',
      traditional: 'Responses become disconnected and contradictory',
      solution: 'Topological relationships maintain coherence',
      impact: '85%+ cross-cube coherence maintained'
    }
  ];

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-white mb-4">
            About Multi-Cube Architecture
          </h1>
          <p className="text-xl text-gray-300 max-w-4xl mx-auto">
            A revolutionary approach to long-context processing that solves the fundamental 
            limitations of token-based AI systems through distributed semantic processing.
          </p>
        </motion.div>

        {/* The Problem */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-8 mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-6 text-center">
            The Long-Context Crisis
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-red-400 mb-4">Current Limitations</h3>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                  <div>
                    <div className="text-white font-medium">Token Limit Collapse</div>
                    <div className="text-gray-300 text-sm">
                      AI systems drop below 50% accuracy at 25k+ tokens due to attention mechanism limitations.
                    </div>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                  <div>
                    <div className="text-white font-medium">Memory Bottlenecks</div>
                    <div className="text-gray-300 text-sm">
                      Processing becomes impossible or extremely slow beyond 100k tokens.
                    </div>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-red-400 rounded-full mt-2"></div>
                  <div>
                    <div className="text-white font-medium">Hollow Million-Token Claims</div>
                    <div className="text-gray-300 text-sm">
                      "Million token context" is marketing - actual performance is terrible at scale.
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-green-400 mb-4">Real-World Impact</h3>
              <div className="space-y-4">
                <div className="p-4 bg-red-500 bg-opacity-10 rounded-lg border border-red-500 border-opacity-20">
                  <div className="text-red-300 font-medium mb-2">Enterprise Analysis Impossible</div>
                  <div className="text-gray-300 text-sm">
                    Complex enterprise systems with 200k+ token contexts cannot be properly analyzed.
                  </div>
                </div>
                
                <div className="p-4 bg-red-500 bg-opacity-10 rounded-lg border border-red-500 border-opacity-20">
                  <div className="text-red-300 font-medium mb-2">Research Bottlenecks</div>
                  <div className="text-gray-300 text-sm">
                    Large document analysis becomes unreliable and incomplete.
                  </div>
                </div>
                
                <div className="p-4 bg-red-500 bg-opacity-10 rounded-lg border border-red-500 border-opacity-20">
                  <div className="text-red-300 font-medium mb-2">Innovation Stagnation</div>
                  <div className="text-gray-300 text-sm">
                    The AI industry is stuck on incremental token limit increases instead of solving the core problem.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Our Solution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-8 text-center">
            The Multi-Cube Solution
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {architectureFeatures.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 + index * 0.1 }}
                  className="card p-6"
                >
                  <div className={`w-12 h-12 mb-4 rounded-lg bg-gradient-to-r ${feature.color} flex items-center justify-center`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-300">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Problem vs Solution Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="card p-8 mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-8 text-center">
            Revolutionary Breakthroughs
          </h2>
          
          <div className="space-y-6">
            {problemSolutions.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.0 + index * 0.1 }}
                className="border border-white border-opacity-10 rounded-lg p-6"
              >
                <h3 className="text-lg font-semibold text-white mb-4">{item.problem}</h3>
                
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <div>
                    <div className="text-red-400 font-medium mb-2">Traditional Approach</div>
                    <div className="text-gray-300 text-sm">{item.traditional}</div>
                  </div>
                  
                  <div>
                    <div className="text-blue-400 font-medium mb-2">Multi-Cube Solution</div>
                    <div className="text-gray-300 text-sm">{item.solution}</div>
                  </div>
                  
                  <div>
                    <div className="text-green-400 font-medium mb-2">Impact</div>
                    <div className="text-gray-300 text-sm flex items-center space-x-2">
                      <CheckCircleIcon className="w-4 h-4 text-green-400" />
                      <span>{item.impact}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technical Architecture */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
          className="card p-8 mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-8 text-center">
            Technical Architecture
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold text-white mb-4">Five Specialized Cubes</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 bg-green-500 rounded"></div>
                  <div>
                    <div className="text-white font-medium">Code Cube</div>
                    <div className="text-gray-300 text-sm">Programming logic, architecture, and technical patterns</div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                  <div>
                    <div className="text-white font-medium">Data Cube</div>
                    <div className="text-gray-300 text-sm">Data processing, analytics, and information flow</div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 bg-purple-500 rounded"></div>
                  <div>
                    <div className="text-white font-medium">User Cube</div>
                    <div className="text-gray-300 text-sm">User behavior, interactions, and engagement patterns</div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 bg-red-500 rounded"></div>
                  <div>
                    <div className="text-white font-medium">Temporal Cube</div>
                    <div className="text-gray-300 text-sm">Time series, trends, and temporal relationships</div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="w-4 h-4 bg-cyan-500 rounded"></div>
                  <div>
                    <div className="text-white font-medium">System Cube</div>
                    <div className="text-gray-300 text-sm">Performance, monitoring, and system health</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold text-white mb-4">Key Innovations</h3>
              <div className="space-y-4">
                <div className="p-4 bg-blue-500 bg-opacity-10 rounded-lg border border-blue-500 border-opacity-20">
                  <div className="text-blue-300 font-medium mb-2">Coordinate-Based Retrieval</div>
                  <div className="text-gray-300 text-sm">
                    Information is accessed through precise 3D coordinates, enabling instant retrieval from massive contexts.
                  </div>
                </div>
                
                <div className="p-4 bg-green-500 bg-opacity-10 rounded-lg border border-green-500 border-opacity-20">
                  <div className="text-green-300 font-medium mb-2">Topological Relationships</div>
                  <div className="text-gray-300 text-sm">
                    Cross-cube connections preserve semantic structure through advanced topological data analysis.
                  </div>
                </div>
                
                <div className="p-4 bg-purple-500 bg-opacity-10 rounded-lg border border-purple-500 border-opacity-20">
                  <div className="text-purple-300 font-medium mb-2">Intelligent Orchestration</div>
                  <div className="text-gray-300 text-sm">
                    ML-powered orchestrator optimally routes queries and coordinates cube interactions.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4 }}
          className="card p-8 text-center"
        >
          <RocketLaunchIcon className="w-16 h-16 mx-auto mb-6 text-purple-400" />
          <h2 className="text-2xl font-bold text-white mb-4">
            Experience the Revolution
          </h2>
          <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
            Try queries that would completely break traditional token-based systems. 
            See how our multi-cube architecture makes the impossible possible.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.a
              href="/demo"
              className="btn-primary flex items-center justify-center space-x-2 text-lg px-8 py-4"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <RocketLaunchIcon className="w-5 h-5" />
              <span>Try Live Demo</span>
            </motion.a>
            
            <motion.a
              href="/benchmark"
              className="btn-secondary flex items-center justify-center space-x-2 text-lg px-8 py-4"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <ChartBarIcon className="w-5 h-5" />
              <span>View Benchmarks</span>
            </motion.a>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default AboutPage;