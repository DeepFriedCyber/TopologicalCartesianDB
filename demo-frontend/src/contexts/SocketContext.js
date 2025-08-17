import React, { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import toast from 'react-hot-toast';

const SocketContext = createContext();

export function SocketProvider({ children }) {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [realTimeData, setRealTimeData] = useState({});

  useEffect(() => {
    // Initialize socket connection
    const socketInstance = io(process.env.REACT_APP_SOCKET_URL || 'http://localhost:5000', {
      transports: ['websocket', 'polling']
    });

    // Connection event handlers
    socketInstance.on('connect', () => {
      console.log('Connected to Multi-Cube Demo Server');
      setConnected(true);
      toast.success('Connected to demo server');
    });

    socketInstance.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
      toast.error('Disconnected from server');
    });

    socketInstance.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setConnected(false);
      toast.error('Failed to connect to server');
    });

    // Demo-specific event handlers
    socketInstance.on('scenario_loading', (data) => {
      toast.loading(`Loading scenario: ${data.total_documents} documents, ~${data.estimated_tokens} tokens`, {
        id: 'scenario-loading'
      });
      
      setRealTimeData(prev => ({
        ...prev,
        scenarioLoading: {
          totalDocuments: data.total_documents,
          estimatedTokens: data.estimated_tokens,
          sessionId: data.session_id
        }
      }));
    });

    socketInstance.on('scenario_loaded', (data) => {
      toast.success('Scenario loaded successfully!', {
        id: 'scenario-loading'
      });
      
      setRealTimeData(prev => ({
        ...prev,
        scenarioLoaded: {
          distributionStats: data.distribution_stats,
          contextSize: data.context_size,
          sessionId: data.session_id
        },
        scenarioLoading: null
      }));
    });

    socketInstance.on('scenario_error', (data) => {
      toast.error(`Failed to load scenario: ${data.error}`, {
        id: 'scenario-loading'
      });
      
      setRealTimeData(prev => ({
        ...prev,
        scenarioError: data.error,
        scenarioLoading: null
      }));
    });

    socketInstance.on('query_processing_started', (data) => {
      toast.loading(`Processing query: "${data.query.substring(0, 50)}..."`, {
        id: 'query-processing'
      });
      
      setRealTimeData(prev => ({
        ...prev,
        queryProcessing: {
          query: data.query,
          estimatedTime: data.estimated_time
        }
      }));
    });

    socketInstance.on('query_result', (data) => {
      toast.success(`Query completed with ${(data.result.accuracy_estimate * 100).toFixed(1)}% accuracy`, {
        id: 'query-processing'
      });
      
      setRealTimeData(prev => ({
        ...prev,
        queryResult: {
          query: data.query,
          result: data.result,
          timestamp: new Date()
        },
        queryProcessing: null
      }));
    });

    socketInstance.on('query_error', (data) => {
      toast.error(`Query failed: ${data.error}`, {
        id: 'query-processing'
      });
      
      setRealTimeData(prev => ({
        ...prev,
        queryError: data.error,
        queryProcessing: null
      }));
    });

    // Performance monitoring events
    socketInstance.on('performance_update', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        performanceUpdate: {
          ...data,
          timestamp: new Date()
        }
      }));
    });

    socketInstance.on('cube_state_update', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        cubeStates: {
          ...prev.cubeStates,
          ...data.cube_states
        }
      }));
    });

    setSocket(socketInstance);

    // Cleanup on unmount
    return () => {
      socketInstance.disconnect();
    };
  }, []);

  // Socket utility functions
  const socketActions = {
    joinSession: (sessionId) => {
      if (socket && connected) {
        socket.emit('join_session', { session_id: sessionId });
      }
    },

    processRealTimeQuery: (sessionId, query) => {
      if (socket && connected) {
        socket.emit('real_time_query', {
          session_id: sessionId,
          query: query
        });
      } else {
        toast.error('Not connected to server');
      }
    },

    requestPerformanceUpdate: (sessionId) => {
      if (socket && connected) {
        socket.emit('request_performance_update', { session_id: sessionId });
      }
    },

    // Subscribe to real-time cube visualization updates
    subscribeToCubeUpdates: (sessionId) => {
      if (socket && connected) {
        socket.emit('subscribe_cube_updates', { session_id: sessionId });
      }
    },

    unsubscribeFromCubeUpdates: (sessionId) => {
      if (socket && connected) {
        socket.emit('unsubscribe_cube_updates', { session_id: sessionId });
      }
    }
  };

  const value = {
    socket,
    connected,
    realTimeData,
    socketActions,
    setRealTimeData
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket() {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
}