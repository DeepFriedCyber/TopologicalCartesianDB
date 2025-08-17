import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  PaperAirplaneIcon, 
  LightBulbIcon,
  CogIcon,
  BoltIcon
} from '@heroicons/react/24/outline';

function QueryInterface({ 
  currentQuery, 
  setCurrentQuery, 
  onProcessQuery, 
  onRealTimeQuery,
  sampleQueries, 
  processing, 
  connected 
}) {
  const [strategy, setStrategy] = useState('adaptive');
  const [useRealTime, setUseRealTime] = useState(false);

  const strategies = {
    adaptive: {
      name: 'Adaptive',
      description: 'Automatically selects optimal strategy',
      icon: CogIcon,
      color: 'from-blue-500 to-cyan-500'
    },
    parallel: {
      name: 'Parallel',
      description: 'Process across multiple cubes simultaneously',
      icon: BoltIcon,
      color: 'from-purple-500 to-pink-500'
    },
    topological: {
      name: 'Topological',
      description: 'Use cross-cube relationships for guidance',
      icon: LightBulbIcon,
      color: 'from-green-500 to-emerald-500'
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!currentQuery.trim() || processing) return;

    if (useRealTime && connected) {
      onRealTimeQuery(currentQuery);
    } else {
      onProcessQuery(currentQuery, strategy);
    }
  };

  const handleSampleQuery = (query) => {
    setCurrentQuery(query);
  };

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-white mb-6 flex items-center space-x-2">
        <PaperAirplaneIcon className="w-5 h-5" />
        <span>Query Interface</span>
      </h3>

      {/* Query Input Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Query Text Area */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Enter your query
          </label>
          <textarea
            value={currentQuery}
            onChange={(e) => setCurrentQuery(e.target.value)}
            placeholder="Ask anything about the loaded context..."
            rows={4}
            className="w-full px-3 py-2 bg-white bg-opacity-10 border border-white border-opacity-20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            disabled={processing}
          />
        </div>

        {/* Strategy Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Processing Strategy
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {Object.entries(strategies).map(([key, strategyInfo]) => {
              const Icon = strategyInfo.icon;
              return (
                <motion.label
                  key={key}
                  className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                    strategy === key
                      ? 'border-blue-500 bg-blue-500 bg-opacity-20'
                      : 'border-white border-opacity-20 bg-white bg-opacity-5 hover:bg-opacity-10'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <input
                    type="radio"
                    name="strategy"
                    value={key}
                    checked={strategy === key}
                    onChange={(e) => setStrategy(e.target.value)}
                    className="sr-only"
                    disabled={processing}
                  />
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4 text-white" />
                    <div>
                      <div className="text-white text-sm font-medium">
                        {strategyInfo.name}
                      </div>
                      <div className="text-gray-400 text-xs">
                        {strategyInfo.description}
                      </div>
                    </div>
                  </div>
                </motion.label>
              );
            })}
          </div>
        </div>

        {/* Real-time Processing Toggle */}
        <div className="flex items-center justify-between p-3 bg-white bg-opacity-5 rounded-lg">
          <div>
            <div className="text-white text-sm font-medium">Real-time Processing</div>
            <div className="text-gray-400 text-xs">
              Use WebSocket for live updates {!connected && '(requires connection)'}
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={useRealTime}
              onChange={(e) => setUseRealTime(e.target.checked)}
              disabled={!connected || processing}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
          </label>
        </div>

        {/* Submit Button */}
        <motion.button
          type="submit"
          disabled={processing || !currentQuery.trim()}
          className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
            processing || !currentQuery.trim()
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white hover:shadow-lg'
          }`}
          whileHover={!processing && currentQuery.trim() ? { scale: 1.02 } : {}}
          whileTap={!processing && currentQuery.trim() ? { scale: 0.98 } : {}}
        >
          {processing ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Processing Query...</span>
            </>
          ) : (
            <>
              <PaperAirplaneIcon className="w-5 h-5" />
              <span>
                {useRealTime ? 'Process Real-time' : 'Process Query'}
              </span>
            </>
          )}
        </motion.button>
      </form>

      {/* Sample Queries */}
      {sampleQueries.length > 0 && (
        <div className="mt-6">
          <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
            <LightBulbIcon className="w-4 h-4" />
            <span>Sample Queries</span>
          </h4>
          <div className="space-y-2">
            {sampleQueries.map((query, index) => (
              <motion.button
                key={index}
                onClick={() => handleSampleQuery(query)}
                disabled={processing}
                className="w-full text-left p-3 bg-white bg-opacity-5 hover:bg-opacity-10 rounded-lg border border-white border-opacity-10 hover:border-opacity-20 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={!processing ? { scale: 1.01 } : {}}
                whileTap={!processing ? { scale: 0.99 } : {}}
              >
                <div className="text-white text-sm">
                  {query}
                </div>
              </motion.button>
            ))}
          </div>
        </div>
      )}

      {/* Query Tips */}
      <div className="mt-6 p-4 bg-blue-500 bg-opacity-10 rounded-lg border border-blue-500 border-opacity-20">
        <h4 className="text-blue-300 font-medium mb-2">💡 Query Tips</h4>
        <ul className="text-blue-200 text-sm space-y-1">
          <li>• Ask complex questions that span multiple domains</li>
          <li>• Try queries that would overwhelm traditional systems</li>
          <li>• Use real-time processing for live updates</li>
          <li>• Experiment with different processing strategies</li>
        </ul>
      </div>
    </div>
  );
}

export default QueryInterface;