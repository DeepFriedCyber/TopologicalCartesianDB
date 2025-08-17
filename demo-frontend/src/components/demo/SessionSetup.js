import React from 'react';
import { motion } from 'framer-motion';
import { PlayIcon, UserIcon } from '@heroicons/react/24/outline';

function SessionSetup({ 
  customerName, 
  setCustomerName, 
  selectedScenario, 
  setSelectedScenario, 
  scenarios, 
  onStartSession, 
  loading 
}) {
  const handleSubmit = (e) => {
    e.preventDefault();
    onStartSession();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6"
    >
      <h3 className="text-lg font-semibold text-white mb-6 flex items-center space-x-2">
        <UserIcon className="w-5 h-5" />
        <span>Start Demo Session</span>
      </h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Customer Name Input */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Your Name
          </label>
          <input
            type="text"
            value={customerName}
            onChange={(e) => setCustomerName(e.target.value)}
            placeholder="Enter your name"
            className="w-full px-3 py-2 bg-white bg-opacity-10 border border-white border-opacity-20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
          />
        </div>

        {/* Scenario Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Demo Scenario
          </label>
          <div className="space-y-2">
            {Object.entries(scenarios).map(([key, scenario]) => (
              <motion.label
                key={key}
                className={`block p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                  selectedScenario === key
                    ? 'border-blue-500 bg-blue-500 bg-opacity-20'
                    : 'border-white border-opacity-20 bg-white bg-opacity-5 hover:bg-opacity-10'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <input
                  type="radio"
                  name="scenario"
                  value={key}
                  checked={selectedScenario === key}
                  onChange={(e) => setSelectedScenario(e.target.value)}
                  className="sr-only"
                />
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-white font-medium">{scenario.name}</div>
                    <div className="text-gray-300 text-sm">{scenario.description}</div>
                  </div>
                  <div className="text-right">
                    <div className={`text-xs px-2 py-1 rounded-full ${
                      scenario.difficulty === 'Easy' ? 'bg-green-500' :
                      scenario.difficulty === 'Medium' ? 'bg-yellow-500' :
                      scenario.difficulty === 'Hard' ? 'bg-red-500' :
                      'bg-purple-500'
                    } text-white`}>
                      {scenario.difficulty}
                    </div>
                    <div className="text-gray-400 text-xs mt-1">
                      ~{(scenario.estimated_tokens / 1000).toFixed(0)}k tokens
                    </div>
                  </div>
                </div>
              </motion.label>
            ))}
          </div>
        </div>

        {/* Start Button */}
        <motion.button
          type="submit"
          disabled={loading || !customerName.trim()}
          className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
            loading || !customerName.trim()
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white hover:shadow-lg'
          }`}
          whileHover={!loading && customerName.trim() ? { scale: 1.02 } : {}}
          whileTap={!loading && customerName.trim() ? { scale: 0.98 } : {}}
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Starting Session...</span>
            </>
          ) : (
            <>
              <PlayIcon className="w-5 h-5" />
              <span>Start Demo Session</span>
            </>
          )}
        </motion.button>
      </form>

      {/* Scenario Preview */}
      {selectedScenario && scenarios[selectedScenario] && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-6 p-4 bg-white bg-opacity-5 rounded-lg border border-white border-opacity-10"
        >
          <h4 className="text-white font-medium mb-2">What to Expect:</h4>
          <div className="text-gray-300 text-sm space-y-1">
            <div>• Context Size: ~{(scenarios[selectedScenario].estimated_tokens / 1000).toFixed(0)}k tokens</div>
            <div>• Difficulty: {scenarios[selectedScenario].difficulty}</div>
            <div>• Traditional systems would {
              scenarios[selectedScenario].estimated_tokens > 100000 ? 'completely fail' :
              scenarios[selectedScenario].estimated_tokens > 25000 ? 'have <50% accuracy' :
              'work normally'
            }</div>
            <div>• Multi-Cube expected accuracy: {
              scenarios[selectedScenario].estimated_tokens > 500000 ? '78-82%' :
              scenarios[selectedScenario].estimated_tokens > 200000 ? '82-88%' :
              scenarios[selectedScenario].estimated_tokens > 50000 ? '88-95%' :
              '95-100%'
            }</div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

export default SessionSetup;