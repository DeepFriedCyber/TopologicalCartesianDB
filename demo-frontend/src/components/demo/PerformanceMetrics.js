import React from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  ClockIcon,
  CubeIcon,
  CheckCircleIcon,
  TrophyIcon
} from '@heroicons/react/24/outline';

function PerformanceMetrics({ sessionStats, cubeUtilization, scenario }) {
  // Debug logging
  console.log('PerformanceMetrics props:', { sessionStats, cubeUtilization, scenario });
  
  const formatUptime = (minutes) => {
    if (minutes < 1) return `${Math.round(minutes * 60)}s`;
    if (minutes < 60) return `${Math.round(minutes)}m`;
    return `${Math.round(minutes / 60)}h ${Math.round(minutes % 60)}m`;
  };

  const getPerformanceStatus = (accuracy) => {
    if (accuracy >= 0.9) return { status: 'Excellent', color: 'text-green-400', bg: 'bg-green-500' };
    if (accuracy >= 0.8) return { status: 'Very Good', color: 'text-blue-400', bg: 'bg-blue-500' };
    if (accuracy >= 0.7) return { status: 'Good', color: 'text-yellow-400', bg: 'bg-yellow-500' };
    return { status: 'Needs Work', color: 'text-red-400', bg: 'bg-red-500' };
  };

  const performanceStatus = getPerformanceStatus(sessionStats.avg_accuracy);

  // Calculate traditional system expected performance
  const getTraditionalPerformance = (contextSize) => {
    if (contextSize <= 10000) return 1.0;
    if (contextSize <= 25000) return 0.85;
    if (contextSize <= 50000) return 0.15;
    if (contextSize <= 100000) return 0.05;
    return 0.0;
  };

  const traditionalAccuracy = getTraditionalPerformance(sessionStats.context_size);

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-white mb-6 flex items-center space-x-2">
        <ChartBarIcon className="w-5 h-5" />
        <span>Performance Metrics</span>
      </h3>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Accuracy */}
        <div className="bg-white bg-opacity-5 rounded-lg p-4 border border-white border-opacity-10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-300 text-sm">Accuracy</span>
            <CheckCircleIcon className={`w-4 h-4 ${performanceStatus.color}`} />
          </div>
          <div className={`text-2xl font-bold ${performanceStatus.color}`}>
            {(sessionStats.avg_accuracy * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {performanceStatus.status}
          </div>
          
          {/* Accuracy Progress Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2 mt-3">
            <motion.div 
              className={`${performanceStatus.bg} h-2 rounded-full`}
              initial={{ width: 0 }}
              animate={{ width: `${sessionStats.avg_accuracy * 100}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
            />
          </div>
        </div>

        {/* Queries Processed */}
        <div className="bg-white bg-opacity-5 rounded-lg p-4 border border-white border-opacity-10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-300 text-sm">Queries</span>
            <CubeIcon className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-blue-400">
            {sessionStats.queries_processed}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Processed
          </div>
        </div>

        {/* Context Size */}
        <div className="bg-white bg-opacity-5 rounded-lg p-4 border border-white border-opacity-10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-300 text-sm">Context</span>
            <div className="text-xs bg-purple-500 bg-opacity-20 text-purple-300 px-2 py-1 rounded">
              {scenario?.difficulty || 'Unknown'}
            </div>
          </div>
          <div className="text-2xl font-bold text-purple-400">
            {(sessionStats.context_size / 1000).toFixed(0)}k
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Tokens
          </div>
        </div>

        {/* Uptime */}
        <div className="bg-white bg-opacity-5 rounded-lg p-4 border border-white border-opacity-10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-300 text-sm">Uptime</span>
            <ClockIcon className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-400">
            {formatUptime(sessionStats.uptime_minutes)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Active
          </div>
        </div>
      </div>

      {/* Performance Comparison */}
      <div className="mb-6">
        <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
          <TrophyIcon className="w-4 h-4" />
          <span>vs Traditional Systems</span>
        </h4>
        
        <div className="space-y-3">
          {/* Multi-Cube Performance */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-green-300">Multi-Cube Architecture</span>
              <span className="text-sm font-semibold text-green-400">
                {(sessionStats.avg_accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <motion.div 
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-3 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${sessionStats.avg_accuracy * 100}%` }}
                transition={{ duration: 1.2, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Traditional System Performance */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-red-300">Traditional Token-Based</span>
              <span className="text-sm font-semibold text-red-400">
                {(traditionalAccuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <motion.div 
                className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${traditionalAccuracy * 100}%` }}
                transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
              />
            </div>
          </div>
        </div>

        {/* Performance Advantage */}
        <div className="mt-4 p-3 bg-gradient-to-r from-green-500 to-blue-500 bg-opacity-10 rounded-lg border border-green-500 border-opacity-20">
          <div className="flex items-center space-x-2">
            <TrophyIcon className="w-5 h-5 text-yellow-400" />
            <div>
              <div className="text-white font-medium">
                {traditionalAccuracy === 0 ? '🚀 Revolutionary Performance' : 
                 sessionStats.avg_accuracy > traditionalAccuracy * 2 ? '⚡ Superior Performance' :
                 '✅ Better Performance'}
              </div>
              <div className="text-green-300 text-sm">
                {traditionalAccuracy === 0 
                  ? 'Traditional systems would completely fail at this scale'
                  : `${((sessionStats.avg_accuracy / traditionalAccuracy - 1) * 100).toFixed(0)}% better than traditional systems`
                }
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Cube Utilization */}
      {cubeUtilization && Object.keys(cubeUtilization).length > 0 && (
        <div>
          <h4 className="text-white font-medium mb-3">Cube Utilization</h4>
          <div className="space-y-2">
            {Object.entries(cubeUtilization).map(([cubeName, utilization]) => (
              <div key={cubeName}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-300 capitalize">
                    {cubeName.replace('_', ' ')}
                  </span>
                  <span className="text-sm font-semibold text-blue-400">
                    {(utilization * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <motion.div 
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${utilization * 100}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Context Scale Indicator */}
      <div className="mt-6 pt-4 border-t border-white border-opacity-20">
        <div className="text-center">
          <div className="text-gray-400 text-xs mb-2">Context Scale Assessment</div>
          {sessionStats.context_size > 500000 ? (
            <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 rounded-full text-sm font-medium">
              🚀 Impossible Scale - Revolutionary
            </div>
          ) : sessionStats.context_size > 200000 ? (
            <div className="bg-gradient-to-r from-red-500 to-orange-500 text-white px-4 py-2 rounded-full text-sm font-medium">
              🔥 Massive Scale - Game Changing
            </div>
          ) : sessionStats.context_size > 50000 ? (
            <div className="bg-gradient-to-r from-yellow-500 to-red-500 text-white px-4 py-2 rounded-full text-sm font-medium">
              ⚡ Large Scale - Advanced
            </div>
          ) : sessionStats.context_size > 25000 ? (
            <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-full text-sm font-medium">
              💪 Medium Scale - Strong
            </div>
          ) : (
            <div className="bg-gradient-to-r from-green-500 to-blue-500 text-white px-4 py-2 rounded-full text-sm font-medium">
              ✅ Standard Scale - Baseline
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default PerformanceMetrics;