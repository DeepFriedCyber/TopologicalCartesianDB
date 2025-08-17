import React from 'react';
import { motion } from 'framer-motion';
import { 
  CubeIcon, 
  LightBulbIcon,
  ExclamationTriangleIcon,
  FireIcon,
  RocketLaunchIcon
} from '@heroicons/react/24/outline';

function ScenarioSelector({ scenarios, selectedScenario, onSelectScenario, sampleQueries }) {
  const getDifficultyIcon = (difficulty) => {
    switch (difficulty) {
      case 'Easy': return CubeIcon;
      case 'Medium': return LightBulbIcon;
      case 'Hard': return ExclamationTriangleIcon;
      case 'Impossible': return RocketLaunchIcon;
      default: return CubeIcon;
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy': return 'from-green-500 to-emerald-500';
      case 'Medium': return 'from-blue-500 to-cyan-500';
      case 'Hard': return 'from-orange-500 to-red-500';
      case 'Impossible': return 'from-purple-500 to-pink-500';
      default: return 'from-gray-500 to-gray-600';
    }
  };

  const getTraditionalSystemStatus = (tokens) => {
    if (tokens <= 10000) return { status: 'Works Fine', color: 'text-green-400', accuracy: '85-100%' };
    if (tokens <= 25000) return { status: 'Degraded', color: 'text-yellow-400', accuracy: '45-85%' };
    if (tokens <= 100000) return { status: 'Mostly Fails', color: 'text-red-400', accuracy: '5-15%' };
    return { status: 'Complete Failure', color: 'text-red-500', accuracy: '0%' };
  };

  const getMultiCubeStatus = (tokens) => {
    if (tokens <= 50000) return { status: 'Excellent', color: 'text-green-400', accuracy: '92-100%' };
    if (tokens <= 200000) return { status: 'Very Good', color: 'text-blue-400', accuracy: '88-95%' };
    if (tokens <= 500000) return { status: 'Good', color: 'text-cyan-400', accuracy: '82-88%' };
    return { status: 'Revolutionary', color: 'text-purple-400', accuracy: '78-85%' };
  };

  return (
    <div className="space-y-6">
      {/* Scenario Overview */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-white mb-6 flex items-center space-x-2">
          <CubeIcon className="w-5 h-5" />
          <span>Demo Scenarios</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(scenarios).map(([key, scenario]) => {
            const Icon = getDifficultyIcon(scenario.difficulty);
            const traditionalStatus = getTraditionalSystemStatus(scenario.estimated_tokens);
            const multiCubeStatus = getMultiCubeStatus(scenario.estimated_tokens);
            
            return (
              <motion.div
                key={key}
                className={`p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                  selectedScenario === key
                    ? 'border-blue-500 bg-blue-500 bg-opacity-20'
                    : 'border-white border-opacity-20 bg-white bg-opacity-5 hover:bg-opacity-10'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onSelectScenario(key)}
              >
                {/* Scenario Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${getDifficultyColor(scenario.difficulty)} flex items-center justify-center`}>
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <div className="text-white font-medium">{scenario.name}</div>
                      <div className="text-gray-300 text-sm">{scenario.description}</div>
                    </div>
                  </div>
                  
                  <div className={`text-xs px-2 py-1 rounded-full ${
                    scenario.difficulty === 'Easy' ? 'bg-green-500' :
                    scenario.difficulty === 'Medium' ? 'bg-blue-500' :
                    scenario.difficulty === 'Hard' ? 'bg-orange-500' :
                    'bg-purple-500'
                  } text-white`}>
                    {scenario.difficulty}
                  </div>
                </div>

                {/* Token Count */}
                <div className="mb-4">
                  <div className="text-gray-400 text-xs">Context Size</div>
                  <div className="text-white font-semibold">
                    ~{(scenario.estimated_tokens / 1000).toFixed(0)}k tokens
                  </div>
                </div>

                {/* Performance Comparison */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Traditional Systems:</span>
                    <span className={`text-xs font-medium ${traditionalStatus.color}`}>
                      {traditionalStatus.status} ({traditionalStatus.accuracy})
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Multi-Cube:</span>
                    <span className={`text-xs font-medium ${multiCubeStatus.color}`}>
                      {multiCubeStatus.status} ({multiCubeStatus.accuracy})
                    </span>
                  </div>
                </div>

                {/* Revolutionary Badge */}
                {scenario.estimated_tokens > 100000 && (
                  <div className="mt-3 p-2 bg-gradient-to-r from-purple-500 to-pink-500 bg-opacity-20 rounded border border-purple-500 border-opacity-30">
                    <div className="text-purple-300 text-xs font-medium flex items-center space-x-1">
                      <RocketLaunchIcon className="w-3 h-3" />
                      <span>Revolutionary Scale</span>
                    </div>
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Sample Queries Preview */}
      {selectedScenario && sampleQueries[selectedScenario] && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-6"
        >
          <h4 className="text-white font-medium mb-4 flex items-center space-x-2">
            <LightBulbIcon className="w-4 h-4" />
            <span>Sample Queries for {scenarios[selectedScenario]?.name}</span>
          </h4>
          
          <div className="space-y-2">
            {sampleQueries[selectedScenario].map((query, index) => (
              <div
                key={index}
                className="p-3 bg-white bg-opacity-5 rounded-lg border border-white border-opacity-10"
              >
                <div className="text-gray-300 text-sm">
                  {query}
                </div>
              </div>
            ))}
          </div>

          {/* Query Complexity Indicator */}
          <div className="mt-4 p-3 bg-blue-500 bg-opacity-10 rounded-lg border border-blue-500 border-opacity-20">
            <div className="text-blue-300 text-sm">
              <strong>Why These Queries Matter:</strong>
              {selectedScenario === 'startup_demo' && (
                <span> These queries test basic multi-domain analysis capabilities that work well in both traditional and multi-cube systems.</span>
              )}
              {selectedScenario === 'enterprise_demo' && (
                <span> These queries require cross-domain correlation that starts to challenge traditional token-based systems.</span>
              )}
              {selectedScenario === 'massive_context_demo' && (
                <span> These queries demand complex reasoning across massive contexts where traditional systems mostly fail.</span>
              )}
              {selectedScenario === 'impossible_context_demo' && (
                <span> These queries are impossible for traditional systems but showcase the revolutionary capability of multi-cube architecture.</span>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Getting Started Guide */}
      <div className="card p-6">
        <h4 className="text-white font-medium mb-4">🚀 Getting Started</h4>
        <div className="space-y-3 text-sm text-gray-300">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">1</div>
            <div>
              <div className="text-white font-medium">Choose Your Challenge</div>
              <div>Select a scenario based on the complexity you want to test</div>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">2</div>
            <div>
              <div className="text-white font-medium">Enter Your Name</div>
              <div>Personalize your demo session for tracking</div>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">3</div>
            <div>
              <div className="text-white font-medium">Start the Demo</div>
              <div>Watch the cubes load and activate in real-time</div>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-bold">4</div>
            <div>
              <div className="text-white font-medium">Ask Complex Questions</div>
              <div>Test queries that would break traditional systems</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ScenarioSelector;