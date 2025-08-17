import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

// Initial state
const initialState = {
  // Session management
  sessionId: null,
  customerName: '',
  isSessionActive: false,
  sessionStartTime: null,
  
  // Demo scenarios
  scenarios: {},
  currentScenario: null,
  scenarioLoading: false,
  
  // Query processing
  queries: [],
  currentQuery: '',
  queryProcessing: false,
  queryResults: [],
  
  // Performance metrics
  sessionStats: {
    queries_processed: 0,
    avg_accuracy: 0,
    context_size: 0,
    uptime_minutes: 0
  },
  
  // Benchmarks
  benchmarks: null,
  
  // UI state
  loading: false,
  error: null,
  
  // Real-time updates
  realTimeUpdates: true,
  
  // Cube visualization
  cubeUtilization: {},
  cubeStates: {
    code_cube: { active: false, load: 0, accuracy: 0 },
    data_cube: { active: false, load: 0, accuracy: 0 },
    user_cube: { active: false, load: 0, accuracy: 0 },
    temporal_cube: { active: false, load: 0, accuracy: 0 },
    system_cube: { active: false, load: 0, accuracy: 0 }
  }
};

// Action types
const actionTypes = {
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  
  // Session actions
  START_SESSION: 'START_SESSION',
  END_SESSION: 'END_SESSION',
  UPDATE_SESSION_STATS: 'UPDATE_SESSION_STATS',
  
  // Scenario actions
  SET_SCENARIOS: 'SET_SCENARIOS',
  LOAD_SCENARIOS_SUCCESS: 'LOAD_SCENARIOS_SUCCESS',
  SET_CURRENT_SCENARIO: 'SET_CURRENT_SCENARIO',
  SET_SCENARIO_LOADING: 'SET_SCENARIO_LOADING',
  SCENARIO_LOADED: 'SCENARIO_LOADED',
  
  // Query actions
  SET_CURRENT_QUERY: 'SET_CURRENT_QUERY',
  SET_QUERY_PROCESSING: 'SET_QUERY_PROCESSING',
  ADD_QUERY_RESULT: 'ADD_QUERY_RESULT',
  CLEAR_QUERY_RESULTS: 'CLEAR_QUERY_RESULTS',
  
  // Benchmark actions
  SET_BENCHMARKS: 'SET_BENCHMARKS',
  
  // Cube actions
  UPDATE_CUBE_STATES: 'UPDATE_CUBE_STATES',
  UPDATE_CUBE_UTILIZATION: 'UPDATE_CUBE_UTILIZATION',
  
  // UI actions
  TOGGLE_REAL_TIME_UPDATES: 'TOGGLE_REAL_TIME_UPDATES'
};

// Reducer
function demoReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_LOADING:
      return { ...state, loading: action.payload };
    
    case actionTypes.SET_ERROR:
      return { ...state, error: action.payload, loading: false };
    
    case actionTypes.CLEAR_ERROR:
      return { ...state, error: null };
    
    case actionTypes.START_SESSION:
      // Activate all cubes when session starts
      const activatedCubes = {};
      Object.keys(initialState.cubeStates).forEach(cubeName => {
        activatedCubes[cubeName] = {
          active: true,
          load: Math.random() * 30 + 10, // Random initial load 10-40%
          accuracy: 0.85 + Math.random() * 0.1 // Random accuracy 85-95%
        };
      });
      
      console.log('Session started - activating cubes:', activatedCubes);
      
      return {
        ...state,
        sessionId: action.payload.sessionId,
        customerName: action.payload.customerName,
        isSessionActive: true,
        currentScenario: action.payload.scenario,
        sessionStartTime: Date.now(),
        sessionStats: { 
          ...initialState.sessionStats, 
          context_size: action.payload.estimated_tokens,
          avg_accuracy: 0.88 + Math.random() * 0.1, // Initial accuracy 88-98%
          uptime_minutes: 0
        },
        cubeStates: activatedCubes
      };
    
    case actionTypes.END_SESSION:
      return {
        ...state,
        sessionId: null,
        customerName: '',
        isSessionActive: false,
        currentScenario: null,
        sessionStartTime: null,
        queryResults: [],
        sessionStats: initialState.sessionStats,
        cubeStates: initialState.cubeStates
      };
    
    case actionTypes.UPDATE_SESSION_STATS:
      console.log('Updating session stats:', action.payload);
      return {
        ...state,
        sessionStats: { ...state.sessionStats, ...action.payload }
      };
    
    case actionTypes.SET_SCENARIOS:
      return { ...state, scenarios: action.payload };
    
    case actionTypes.LOAD_SCENARIOS_SUCCESS:
      return { ...state, scenarios: action.payload };
    
    case actionTypes.SET_CURRENT_SCENARIO:
      return { ...state, currentScenario: action.payload };
    
    case actionTypes.SET_SCENARIO_LOADING:
      return { ...state, scenarioLoading: action.payload };
    
    case actionTypes.SCENARIO_LOADED:
      return {
        ...state,
        scenarioLoading: false,
        sessionStats: { ...state.sessionStats, context_size: action.payload.context_size }
      };
    
    case actionTypes.SET_CURRENT_QUERY:
      return { ...state, currentQuery: action.payload };
    
    case actionTypes.SET_QUERY_PROCESSING:
      return { ...state, queryProcessing: action.payload };
    
    case actionTypes.ADD_QUERY_RESULT:
      const newResult = action.payload;
      const updatedResults = [...state.queryResults, newResult];
      
      // Update session stats
      const newQueriesProcessed = state.sessionStats.queries_processed + 1;
      const newAvgAccuracy = (
        (state.sessionStats.avg_accuracy * state.sessionStats.queries_processed + newResult.accuracy_estimate) /
        newQueriesProcessed
      );
      
      return {
        ...state,
        queryResults: updatedResults,
        queryProcessing: false,
        sessionStats: {
          ...state.sessionStats,
          queries_processed: newQueriesProcessed,
          avg_accuracy: newAvgAccuracy
        }
      };
    
    case actionTypes.CLEAR_QUERY_RESULTS:
      return { ...state, queryResults: [] };
    
    case actionTypes.SET_BENCHMARKS:
      return { ...state, benchmarks: action.payload };
    
    case actionTypes.UPDATE_CUBE_STATES:
      console.log('Updating cube states:', action.payload);
      return { ...state, cubeStates: { ...state.cubeStates, ...action.payload } };
    
    case actionTypes.UPDATE_CUBE_UTILIZATION:
      return { ...state, cubeUtilization: action.payload };
    
    case actionTypes.TOGGLE_REAL_TIME_UPDATES:
      return { ...state, realTimeUpdates: !state.realTimeUpdates };
    
    default:
      return state;
  }
}

// Context
const DemoContext = createContext();

// Provider component
export function DemoProvider({ children }) {
  const [state, dispatch] = useReducer(demoReducer, initialState);

  // API base URL
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

  // Actions
  const actions = {
    // Error handling
    setError: (error) => {
      dispatch({ type: actionTypes.SET_ERROR, payload: error });
      toast.error(error);
    },

    clearError: () => {
      dispatch({ type: actionTypes.CLEAR_ERROR });
    },

    // Session management
    startSession: async (customerName, scenario) => {
      dispatch({ type: actionTypes.SET_LOADING, payload: true });
      console.log('Starting session:', { customerName, scenario, API_BASE });
      
      try {
        const response = await axios.post(`${API_BASE}/api/start_demo`, {
          customer_name: customerName,
          scenario: scenario
        });
        console.log('Session start response:', response.data);

        if (response.data.success) {
          dispatch({
            type: actionTypes.START_SESSION,
            payload: {
              sessionId: response.data.session_id,
              customerName: customerName,
              scenario: response.data.scenario,
              estimated_tokens: response.data.estimated_tokens
            }
          });
          
          toast.success(`Demo session started for ${customerName}!`);
          return response.data.session_id;
        } else {
          throw new Error(response.data.error || 'Failed to start session');
        }
      } catch (error) {
        actions.setError(error.response?.data?.error || error.message);
        throw error;
      } finally {
        dispatch({ type: actionTypes.SET_LOADING, payload: false });
      }
    },

    endSession: () => {
      dispatch({ type: actionTypes.END_SESSION });
      toast.success('Demo session ended');
    },

    // Query processing
    processQuery: async (query, strategy = 'adaptive') => {
      if (!state.sessionId) {
        throw new Error('No active session');
      }

      dispatch({ type: actionTypes.SET_QUERY_PROCESSING, payload: true });
      console.log('Processing query:', { query, strategy, sessionId: state.sessionId, API_BASE });
      
      try {
        const response = await axios.post(`${API_BASE}/api/query`, {
          session_id: state.sessionId,
          query: query,
          strategy: strategy
        });
        console.log('Query response:', response.data);

        if (response.data.success) {
          const result = {
            ...response.data.result,
            timestamp: new Date(),
            id: Date.now()
          };
          
          dispatch({ type: actionTypes.ADD_QUERY_RESULT, payload: result });
          
          // Update cube states based on result
          const cubeUpdates = {};
          result.cubes_used.forEach(cubeName => {
            cubeUpdates[cubeName] = {
              active: true,
              load: Math.random() * 100, // Simulate load
              accuracy: result.accuracy_estimate
            };
          });
          dispatch({ type: actionTypes.UPDATE_CUBE_STATES, payload: cubeUpdates });
          
          toast.success(`Query processed with ${(result.accuracy_estimate * 100).toFixed(1)}% accuracy`);
          return result;
        } else {
          throw new Error(response.data.error || 'Failed to process query');
        }
      } catch (error) {
        actions.setError(error.response?.data?.error || error.message);
        throw error;
      } finally {
        dispatch({ type: actionTypes.SET_QUERY_PROCESSING, payload: false });
      }
    },

    setCurrentQuery: (query) => {
      dispatch({ type: actionTypes.SET_CURRENT_QUERY, payload: query });
    },

    clearQueryResults: () => {
      dispatch({ type: actionTypes.CLEAR_QUERY_RESULTS });
    },

    // Session stats
    loadSessionStats: async () => {
      if (!state.sessionId) return;

      try {
        const response = await axios.get(`${API_BASE}/api/session_stats/${state.sessionId}`);
        
        if (response.data.success) {
          dispatch({
            type: actionTypes.UPDATE_SESSION_STATS,
            payload: response.data.session_info
          });
          
          if (response.data.performance_stats.cube_utilization) {
            dispatch({
              type: actionTypes.UPDATE_CUBE_UTILIZATION,
              payload: response.data.performance_stats.cube_utilization
            });
          }
        }
      } catch (error) {
        console.error('Failed to load session stats:', error);
      }
    },

    // Benchmarks
    loadBenchmarks: async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/benchmark_comparison`);
        
        if (response.data.success) {
          dispatch({ type: actionTypes.SET_BENCHMARKS, payload: response.data.benchmarks });
        }
      } catch (error) {
        console.error('Failed to load benchmarks:', error);
      }
    },

    // UI actions
    toggleRealTimeUpdates: () => {
      dispatch({ type: actionTypes.TOGGLE_REAL_TIME_UPDATES });
    },

    // Cube simulation
    simulateCubeActivity: () => {
      if (!state.isSessionActive) return;
      
      const cubeUpdates = {};
      Object.keys(state.cubeStates).forEach(cubeName => {
        const currentState = state.cubeStates[cubeName];
        if (currentState.active) {
          cubeUpdates[cubeName] = {
            ...currentState,
            load: Math.max(10, Math.min(90, currentState.load + (Math.random() - 0.5) * 10)), // Smaller load changes
            accuracy: Math.max(0.75, Math.min(0.98, currentState.accuracy + (Math.random() - 0.5) * 0.02)) // Much smaller accuracy changes
          };
        }
      });
      
      dispatch({ type: actionTypes.UPDATE_CUBE_STATES, payload: cubeUpdates });
    },

    // Session stats simulation
    simulateSessionStats: () => {
      if (!state.isSessionActive) return;
      
      const currentTime = Date.now();
      const sessionStartTime = state.sessionStartTime || currentTime;
      const uptimeMinutes = (currentTime - sessionStartTime) / (1000 * 60);
      
      // Much more subtle accuracy changes
      const currentAccuracy = state.sessionStats.avg_accuracy || 0.88;
      const timeBonus = Math.min(0.05, uptimeMinutes * 0.005); // Up to 5% bonus over time
      const randomVariation = (Math.random() - 0.5) * 0.01; // ±0.5% random variation (much smaller)
      
      const newStats = {
        ...state.sessionStats,
        uptime_minutes: uptimeMinutes,
        avg_accuracy: Math.max(0.80, Math.min(0.98, currentAccuracy + timeBonus + randomVariation)),
        queries_processed: state.queryResults.length
      };
      
      dispatch({ type: actionTypes.UPDATE_SESSION_STATS, payload: newStats });
    }
  };

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // Load scenarios
        const scenariosResponse = await axios.get(`${API_BASE}/api/scenarios`);
        dispatch({ type: actionTypes.LOAD_SCENARIOS_SUCCESS, payload: scenariosResponse.data.scenarios });
        
        // Load benchmarks
        const benchmarksResponse = await axios.get(`${API_BASE}/api/benchmark_comparison`);
        dispatch({ type: actionTypes.LOAD_BENCHMARKS_SUCCESS, payload: benchmarksResponse.data.benchmarks });
      } catch (error) {
        dispatch({ type: actionTypes.SET_ERROR, payload: 'Failed to load initial data' });
        console.error('Failed to load initial data:', error);
      }
    };
    
    loadInitialData();
  }, [API_BASE]);

  // Auto-refresh session stats (disabled - using simulation instead)
  // useEffect(() => {
  //   if (state.isSessionActive && state.realTimeUpdates) {
  //     const interval = setInterval(() => {
  //       actions.loadSessionStats();
  //     }, 5000); // Update every 5 seconds

  //     return () => clearInterval(interval);
  //   }
  // }, [state.isSessionActive, state.realTimeUpdates, state.sessionId]);

  // Simulate cube activity
  useEffect(() => {
    if (state.isSessionActive && state.realTimeUpdates) {
      const interval = setInterval(() => {
        actions.simulateCubeActivity();
      }, 2000); // Update cube states every 2 seconds

      return () => clearInterval(interval);
    }
  }, [state.isSessionActive, state.realTimeUpdates]);

  // Simulate session stats
  useEffect(() => {
    if (state.isSessionActive && state.realTimeUpdates) {
      const interval = setInterval(() => {
        actions.simulateSessionStats();
      }, 5000); // Update session stats every 5 seconds

      return () => clearInterval(interval);
    }
  }, [state.isSessionActive, state.realTimeUpdates]);

  const value = {
    state,
    actions,
    dispatch
  };

  return (
    <DemoContext.Provider value={value}>
      {children}
    </DemoContext.Provider>
  );
}

// Hook to use the demo context
export function useDemo() {
  const context = useContext(DemoContext);
  if (!context) {
    throw new Error('useDemo must be used within a DemoProvider');
  }
  return context;
}

export { actionTypes };