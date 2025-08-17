import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircleIcon, 
  ClockIcon,
  CubeIcon,
  ChartBarIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

function ResultsDisplay({ queryResults, processing, realTimeData }) {
  const formatTime = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.9) return 'text-green-400';
    if (accuracy >= 0.8) return 'text-blue-400';
    if (accuracy >= 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getAccuracyStatus = (accuracy) => {
    if (accuracy >= 0.9) return 'Excellent';
    if (accuracy >= 0.8) return 'Very Good';
    if (accuracy >= 0.7) return 'Good';
    return 'Needs Improvement';
  };

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-white mb-6 flex items-center space-x-2">
        <ChartBarIcon className="w-5 h-5" />
        <span>Query Results</span>
        {processing && (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 ml-2"></div>
        )}
      </h3>

      {/* Processing Indicator */}
      <AnimatePresence>
        {processing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-6 p-4 bg-blue-500 bg-opacity-20 rounded-lg border border-blue-500 border-opacity-30"
          >
            <div className="flex items-center space-x-3">
              <div className="animate-pulse rounded-full h-3 w-3 bg-blue-500"></div>
              <span className="text-blue-300">Processing your query...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Real-time Processing Status */}
      <AnimatePresence>
        {realTimeData.queryProcessing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-6 p-4 bg-purple-500 bg-opacity-20 rounded-lg border border-purple-500 border-opacity-30"
          >
            <div className="flex items-center space-x-3">
              <div className="animate-bounce rounded-full h-3 w-3 bg-purple-500"></div>
              <span className="text-purple-300">Real-time processing active...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results List */}
      <div className="space-y-4 max-h-96 overflow-y-auto">
        <AnimatePresence>
          {queryResults.length === 0 && !processing ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-8 text-gray-400"
            >
              <CubeIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No queries processed yet.</p>
              <p className="text-sm mt-2">Enter a query above to see results here.</p>
            </motion.div>
          ) : (
            queryResults.map((result, index) => (
              <motion.div
                key={result.id || index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white bg-opacity-5 rounded-lg p-4 border border-white border-opacity-10"
              >
                {/* Query Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="text-white font-medium mb-1">
                      Query #{queryResults.length - index}
                    </div>
                    <div className="text-gray-300 text-sm">
                      {result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : 'Just now'}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircleIcon className="w-5 h-5 text-green-400" />
                    <span className={`text-sm font-medium ${getAccuracyColor(result.accuracy_estimate)}`}>
                      {(result.accuracy_estimate * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Response */}
                <div className="mb-4">
                  <div className="text-white bg-gray-800 bg-opacity-50 rounded-lg p-3">
                    {result.response}
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div className="text-center">
                    <div className="text-gray-400 text-xs">Accuracy</div>
                    <div className={`font-semibold ${getAccuracyColor(result.accuracy_estimate)}`}>
                      {(result.accuracy_estimate * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500">
                      {getAccuracyStatus(result.accuracy_estimate)}
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-gray-400 text-xs">Processing Time</div>
                    <div className="text-blue-400 font-semibold">
                      {formatTime(result.processing_time_ms)}
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-gray-400 text-xs">Strategy</div>
                    <div className="text-purple-400 font-semibold capitalize">
                      {result.strategy_used}
                    </div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-gray-400 text-xs">Context Size</div>
                    <div className="text-cyan-400 font-semibold">
                      {(result.context_tokens_processed / 1000).toFixed(0)}k
                    </div>
                  </div>
                </div>

                {/* Cubes Used */}
                <div className="mb-4">
                  <div className="text-gray-400 text-xs mb-2">Cubes Activated:</div>
                  <div className="flex flex-wrap gap-2">
                    {result.cubes_used.map((cube, cubeIndex) => (
                      <span
                        key={cubeIndex}
                        className="px-2 py-1 bg-blue-500 bg-opacity-20 text-blue-300 text-xs rounded-full border border-blue-500 border-opacity-30"
                      >
                        {cube.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Traditional System Comparison */}
                {result.traditional_system_comparison && (
                  <div className="border-t border-white border-opacity-10 pt-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-gray-400 text-xs">Traditional System Impact:</div>
                        <div className="flex items-center space-x-2 mt-1">
                          {result.traditional_system_comparison.would_fail ? (
                            <>
                              <ExclamationTriangleIcon className="w-4 h-4 text-red-400" />
                              <span className="text-red-400 text-sm">Would Fail</span>
                            </>
                          ) : (
                            <>
                              <CheckCircleIcon className="w-4 h-4 text-green-400" />
                              <span className="text-green-400 text-sm">Would Work</span>
                            </>
                          )}
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-gray-400 text-xs">Expected Accuracy:</div>
                        <div className={`font-semibold ${
                          result.traditional_system_comparison.expected_accuracy > 0.5 
                            ? 'text-yellow-400' 
                            : 'text-red-400'
                        }`}>
                          {(result.traditional_system_comparison.expected_accuracy * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    
                    {result.traditional_system_comparison.would_fail && (
                      <div className="mt-2 p-2 bg-red-500 bg-opacity-10 rounded border border-red-500 border-opacity-20">
                        <div className="text-red-300 text-xs">
                          🚀 <strong>Revolutionary:</strong> This query would completely break traditional token-based systems, 
                          but our multi-cube architecture handles it with {(result.accuracy_estimate * 100).toFixed(1)}% accuracy!
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Cross-Cube Coherence */}
                {result.cross_cube_coherence && (
                  <div className="mt-4 pt-4 border-t border-white border-opacity-10">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400 text-xs">Cross-Cube Coherence:</span>
                      <span className="text-green-400 font-semibold">
                        {(result.cross_cube_coherence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${result.cross_cube_coherence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Summary Stats */}
      {queryResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 pt-6 border-t border-white border-opacity-20"
        >
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-gray-400 text-xs">Total Queries</div>
              <div className="text-white font-semibold text-lg">{queryResults.length}</div>
            </div>
            <div>
              <div className="text-gray-400 text-xs">Avg Accuracy</div>
              <div className="text-green-400 font-semibold text-lg">
                {queryResults.length > 0 
                  ? (queryResults.reduce((sum, r) => sum + r.accuracy_estimate, 0) / queryResults.length * 100).toFixed(1)
                  : 0}%
              </div>
            </div>
            <div>
              <div className="text-gray-400 text-xs">Avg Time</div>
              <div className="text-blue-400 font-semibold text-lg">
                {queryResults.length > 0 
                  ? formatTime(queryResults.reduce((sum, r) => sum + r.processing_time_ms, 0) / queryResults.length)
                  : '0ms'}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default ResultsDisplay;