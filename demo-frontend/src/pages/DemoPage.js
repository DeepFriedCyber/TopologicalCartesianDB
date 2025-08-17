import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  PlayIcon, 
  StopIcon,
  CubeIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useDemo } from '../contexts/DemoContext';
import { useSocket } from '../contexts/SocketContext';
import toast from 'react-hot-toast';

// Components
import SessionSetup from '../components/demo/SessionSetup';
import CubeVisualization from '../components/demo/CubeVisualization';
import QueryInterface from '../components/demo/QueryInterface';
import ResultsDisplay from '../components/demo/ResultsDisplay';
import PerformanceMetrics from '../components/demo/PerformanceMetrics';
import ScenarioSelector from '../components/demo/ScenarioSelector';

function DemoPage() {
  const { state, actions } = useDemo();
  const { connected, socketActions, realTimeData } = useSocket();
  const [selectedScenario, setSelectedScenario] = useState('startup_demo');
  const [customerName, setCustomerName] = useState('');

  // Use scenarios from backend
  const scenarios = state.scenarios || {};

  // Extract sample queries from scenarios
  const sampleQueries = {};
  Object.keys(scenarios).forEach(key => {
    sampleQueries[key] = scenarios[key]?.sample_queries || [];
  });

  // Start demo session
  const handleStartSession = async () => {
    if (!customerName.trim()) {
      toast.error('Please enter your name');
      return;
    }

    try {
      const sessionId = await actions.startSession(customerName, selectedScenario);
      if (sessionId && connected) {
        socketActions.joinSession(sessionId);
        socketActions.subscribeToCubeUpdates(sessionId);
      }
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  // End demo session
  const handleEndSession = () => {
    if (state.sessionId && connected) {
      socketActions.unsubscribeFromCubeUpdates(state.sessionId);
    }
    actions.endSession();
  };

  // Process query
  const handleProcessQuery = async (query, strategy = 'adaptive') => {
    try {
      await actions.processQuery(query, strategy);
    } catch (error) {
      console.error('Failed to process query:', error);
    }
  };

  // Real-time query processing
  const handleRealTimeQuery = (query) => {
    if (state.sessionId && connected) {
      socketActions.processRealTimeQuery(state.sessionId, query);
    } else {
      toast.error('Real-time processing requires active connection');
    }
  };

  // Update cube states from real-time data
  useEffect(() => {
    if (realTimeData.cubeStates) {
      // Update cube visualization with real-time data
    }
  }, [realTimeData.cubeStates]);

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-4">
            Multi-Cube Architecture Live Demo
          </h1>
          <p className="text-xl text-gray-300">
            Experience long-context processing that breaks traditional token limits
          </p>
        </motion.div>

        {/* Connection Status */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-6"
        >
          <div className={`card p-4 ${connected ? 'border-green-500' : 'border-red-500'} border-2`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-white font-medium">
                  {connected ? 'Connected to Multi-Cube Server' : 'Disconnected from Server'}
                </span>
              </div>
              
              {state.isSessionActive && (
                <div className="flex items-center space-x-4 text-sm text-gray-300">
                  <span>Session: {state.sessionId}</span>
                  <span>Context: ~{(state.sessionStats.context_size / 1000).toFixed(0)}k tokens</span>
                  <span>Queries: {state.sessionStats.queries_processed}</span>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Session Setup and Controls */}
          <div className="lg:col-span-1 space-y-6">
            {!state.isSessionActive ? (
              <SessionSetup
                customerName={customerName}
                setCustomerName={setCustomerName}
                selectedScenario={selectedScenario}
                setSelectedScenario={setSelectedScenario}
                scenarios={scenarios}
                onStartSession={handleStartSession}
                loading={state.loading}
              />
            ) : (
              <div className="space-y-6">
                {/* Session Info */}
                <div className="card p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Active Session</h3>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Customer:</span>
                      <span className="text-white">{state.customerName}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Scenario:</span>
                      <span className="text-white">{scenarios[state.currentScenario]?.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Context Size:</span>
                      <span className="text-white">~{(state.sessionStats.context_size / 1000).toFixed(0)}k tokens</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Accuracy:</span>
                      <span className="text-green-400">{(state.sessionStats.avg_accuracy * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={handleEndSession}
                    className="w-full mt-4 bg-red-500 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                  >
                    <StopIcon className="w-4 h-4" />
                    <span>End Session</span>
                  </button>
                </div>

                {/* Performance Metrics */}
                <PerformanceMetrics
                  sessionStats={state.sessionStats}
                  cubeUtilization={state.cubeUtilization}
                  scenario={scenarios[state.currentScenario]}
                />
              </div>
            )}

            {/* Cube Visualization */}
            <CubeVisualization
              cubeStates={state.cubeStates}
              isActive={state.isSessionActive}
              realTimeData={realTimeData}
            />
          </div>

          {/* Right Column - Query Interface and Results */}
          <div className="lg:col-span-2 space-y-6">
            {state.isSessionActive && (
              <>
                {/* Query Interface */}
                <QueryInterface
                  currentQuery={state.currentQuery}
                  setCurrentQuery={actions.setCurrentQuery}
                  onProcessQuery={handleProcessQuery}
                  onRealTimeQuery={handleRealTimeQuery}
                  sampleQueries={sampleQueries[state.currentScenario] || []}
                  processing={state.queryProcessing}
                  connected={connected}
                />

                {/* Results Display */}
                <ResultsDisplay
                  queryResults={state.queryResults}
                  processing={state.queryProcessing}
                  realTimeData={realTimeData}
                />
              </>
            )}

            {/* Scenario Information */}
            {!state.isSessionActive && (
              <ScenarioSelector
                scenarios={scenarios}
                selectedScenario={selectedScenario}
                onSelectScenario={setSelectedScenario}
                sampleQueries={sampleQueries}
              />
            )}
          </div>
        </div>

        {/* Real-time Updates Display */}
        <AnimatePresence>
          {realTimeData.scenarioLoading && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              className="fixed bottom-4 right-4 card p-4 max-w-sm"
            >
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                <div>
                  <p className="text-white font-medium">Loading Scenario</p>
                  <p className="text-gray-300 text-sm">
                    {realTimeData.scenarioLoading.totalDocuments} documents, 
                    ~{realTimeData.scenarioLoading.estimatedTokens} tokens
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {realTimeData.queryProcessing && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              className="fixed bottom-4 right-4 card p-4 max-w-sm"
            >
              <div className="flex items-center space-x-3">
                <div className="animate-pulse rounded-full h-6 w-6 bg-purple-500"></div>
                <div>
                  <p className="text-white font-medium">Processing Query</p>
                  <p className="text-gray-300 text-sm">
                    {realTimeData.queryProcessing.estimatedTime}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default DemoPage;