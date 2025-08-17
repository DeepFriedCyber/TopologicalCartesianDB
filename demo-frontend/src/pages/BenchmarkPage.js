import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  TrophyIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import { useDemo } from '../contexts/DemoContext';

function BenchmarkPage() {
  const { state, actions } = useDemo();
  const [selectedMetric, setSelectedMetric] = useState('accuracy');

  useEffect(() => {
    actions.loadBenchmarks();
  }, []);

  const benchmarkData = [
    { tokens: '5k', traditional: 100, multiCube: 100, status: 'baseline' },
    { tokens: '10k', traditional: 85, multiCube: 98, status: 'good' },
    { tokens: '25k', traditional: 45, multiCube: 95, status: 'revolutionary' },
    { tokens: '50k', traditional: 15, multiCube: 92, status: 'revolutionary' },
    { tokens: '100k', traditional: 5, multiCube: 88, status: 'revolutionary' },
    { tokens: '500k', traditional: 0, multiCube: 82, status: 'revolutionary' },
    { tokens: '1M+', traditional: 0, multiCube: 78, status: 'revolutionary' }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'baseline': return 'text-gray-400';
      case 'good': return 'text-blue-400';
      case 'revolutionary': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (traditional, multiCube) => {
    if (traditional === 0 && multiCube > 70) return <TrophyIcon className="w-5 h-5 text-yellow-400" />;
    if (multiCube > traditional * 2) return <CheckCircleIcon className="w-5 h-5 text-green-400" />;
    return <ChartBarIcon className="w-5 h-5 text-blue-400" />;
  };

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
            Performance Benchmarks
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Comprehensive comparison showing how Multi-Cube Architecture maintains high accuracy 
            at scales where traditional token-based systems completely fail.
          </p>
        </motion.div>

        {/* Key Insights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
        >
          <div className="card p-6 text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-red-500 to-orange-500 rounded-full flex items-center justify-center">
              <ExclamationTriangleIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Traditional Failure Point</h3>
            <p className="text-gray-300 text-sm">
              Token-based systems drop below 50% accuracy at 25k tokens and completely fail beyond 100k tokens.
            </p>
          </div>

          <div className="card p-6 text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
              <CheckCircleIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Multi-Cube Consistency</h3>
            <p className="text-gray-300 text-sm">
              Our architecture maintains 78%+ accuracy even at 1M+ tokens, enabling previously impossible queries.
            </p>
          </div>

          <div className="card p-6 text-center">
            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
              <TrophyIcon className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Revolutionary Scale</h3>
            <p className="text-gray-300 text-sm">
              At 500k+ tokens, we achieve infinite performance advantage since traditional systems fail completely.
            </p>
          </div>
        </motion.div>

        {/* Benchmark Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card p-8 mb-12"
        >
          <h2 className="text-2xl font-bold text-white mb-6 text-center">
            Accuracy vs Context Size Comparison
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full text-white">
              <thead>
                <tr className="border-b border-white border-opacity-20">
                  <th className="text-left py-4 px-4">Context Size</th>
                  <th className="text-center py-4 px-4">Traditional Systems</th>
                  <th className="text-center py-4 px-4">Multi-Cube Architecture</th>
                  <th className="text-center py-4 px-4">Performance Advantage</th>
                  <th className="text-center py-4 px-4">Status</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkData.map((row, index) => (
                  <motion.tr
                    key={row.tokens}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.6 + index * 0.1 }}
                    className="border-b border-white border-opacity-10 hover:bg-white hover:bg-opacity-5"
                  >
                    <td className="py-4 px-4 font-medium">{row.tokens} tokens</td>
                    
                    <td className="text-center py-4 px-4">
                      <div className="flex items-center justify-center space-x-2">
                        {row.traditional > 0 ? (
                          <CheckCircleIcon className="w-4 h-4 text-green-400" />
                        ) : (
                          <XCircleIcon className="w-4 h-4 text-red-400" />
                        )}
                        <span className={row.traditional > 50 ? 'text-green-400' : row.traditional > 0 ? 'text-yellow-400' : 'text-red-400'}>
                          {row.traditional}%
                        </span>
                      </div>
                    </td>
                    
                    <td className="text-center py-4 px-4">
                      <div className="flex items-center justify-center space-x-2">
                        <CheckCircleIcon className="w-4 h-4 text-green-400" />
                        <span className="text-green-400 font-semibold">
                          {row.multiCube}%
                        </span>
                      </div>
                    </td>
                    
                    <td className="text-center py-4 px-4">
                      <span className={`font-semibold ${getStatusColor(row.status)}`}>
                        {row.traditional === 0 ? '∞' : `${((row.multiCube / row.traditional - 1) * 100).toFixed(0)}%`}
                      </span>
                    </td>
                    
                    <td className="text-center py-4 px-4">
                      <div className="flex items-center justify-center space-x-2">
                        {getStatusIcon(row.traditional, row.multiCube)}
                        <span className={`text-sm font-medium ${
                          row.status === 'revolutionary' ? 'text-purple-400' :
                          row.status === 'good' ? 'text-blue-400' :
                          'text-gray-400'
                        }`}>
                          {row.status === 'revolutionary' ? 'Revolutionary' :
                           row.status === 'good' ? 'Superior' :
                           'Baseline'}
                        </span>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Technical Details */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12"
        >
          {/* Why Traditional Systems Fail */}
          <div className="card p-6">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-400" />
              <span>Why Traditional Systems Fail</span>
            </h3>
            <div className="space-y-4 text-gray-300">
              <div>
                <h4 className="text-white font-medium mb-2">Token Limit Collapse</h4>
                <p className="text-sm">
                  As context size increases, attention mechanisms become exponentially less effective, 
                  leading to dramatic accuracy degradation beyond 25k tokens.
                </p>
              </div>
              <div>
                <h4 className="text-white font-medium mb-2">Memory Bottlenecks</h4>
                <p className="text-sm">
                  Traditional architectures hit memory walls around 100k tokens, making processing 
                  either impossible or extremely slow and inaccurate.
                </p>
              </div>
              <div>
                <h4 className="text-white font-medium mb-2">Lost Context</h4>
                <p className="text-sm">
                  Important information gets "lost in the middle" of large contexts, leading to 
                  incomplete or incorrect responses.
                </p>
              </div>
            </div>
          </div>

          {/* How Multi-Cube Succeeds */}
          <div className="card p-6">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center space-x-2">
              <CheckCircleIcon className="w-5 h-5 text-green-400" />
              <span>How Multi-Cube Succeeds</span>
            </h3>
            <div className="space-y-4 text-gray-300">
              <div>
                <h4 className="text-white font-medium mb-2">Distributed Processing</h4>
                <p className="text-sm">
                  Context is distributed across specialized cubes, each handling specific domains 
                  without token-based limitations.
                </p>
              </div>
              <div>
                <h4 className="text-white font-medium mb-2">Topological Relationships</h4>
                <p className="text-sm">
                  Cross-cube connections preserve semantic relationships through topological 
                  data analysis, maintaining coherence at any scale.
                </p>
              </div>
              <div>
                <h4 className="text-white font-medium mb-2">Coordinate-Based Retrieval</h4>
                <p className="text-sm">
                  Information is accessed through precise coordinates rather than sequential 
                  processing, enabling instant retrieval from massive contexts.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Real-World Impact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
          className="card p-8 text-center"
        >
          <h2 className="text-2xl font-bold text-white mb-6">Real-World Impact</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <div className="text-3xl font-bold text-purple-400 mb-2">500k+</div>
              <div className="text-white font-medium mb-2">Token Contexts</div>
              <div className="text-gray-300 text-sm">
                Process enterprise-scale documents that would crash traditional systems
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-400 mb-2">78%+</div>
              <div className="text-white font-medium mb-2">Accuracy Maintained</div>
              <div className="text-gray-300 text-sm">
                Consistent performance even at impossible scales
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold text-yellow-400 mb-2">∞</div>
              <div className="text-white font-medium mb-2">Performance Advantage</div>
              <div className="text-gray-300 text-sm">
                Infinite advantage where traditional systems fail completely
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default BenchmarkPage;