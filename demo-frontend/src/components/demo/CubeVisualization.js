import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

function CubeVisualization({ cubeStates, isActive, realTimeData }) {
  const canvasRef = useRef(null);
  
  // Debug logging
  useEffect(() => {
    console.log('CubeVisualization props:', { cubeStates, isActive, realTimeData });
  }, [cubeStates, isActive, realTimeData]);

  const cubeConfig = {
    code_cube: { 
      color: '#10b981', 
      name: 'Code Cube',
      position: { x: 0.15, y: 0.25 },
      description: 'Programming & Architecture'
    },
    data_cube: { 
      color: '#f59e0b', 
      name: 'Data Cube',
      position: { x: 0.85, y: 0.25 },
      description: 'Data Processing & Analytics'
    },
    user_cube: { 
      color: '#8b5cf6', 
      name: 'User Cube',
      position: { x: 0.15, y: 0.75 },
      description: 'User Behavior & Patterns'
    },
    temporal_cube: { 
      color: '#ef4444', 
      name: 'Temporal Cube',
      position: { x: 0.85, y: 0.75 },
      description: 'Time Series & Trends'
    },
    system_cube: { 
      color: '#06b6d4', 
      name: 'System Cube',
      position: { x: 0.5, y: 0.5 },
      description: 'Performance & Monitoring'
    }
  };

  // Canvas animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    let animationId;
    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections between active cubes
      const activeCubes = Object.entries(cubeStates).filter(([_, state]) => state.active);
      
      if (activeCubes.length > 1) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < activeCubes.length; i++) {
          for (let j = i + 1; j < activeCubes.length; j++) {
            const cube1 = cubeConfig[activeCubes[i][0]];
            const cube2 = cubeConfig[activeCubes[j][0]];
            
            if (cube1 && cube2) {
              const x1 = cube1.position.x * rect.width;
              const y1 = cube1.position.y * rect.height;
              const x2 = cube2.position.x * rect.width;
              const y2 = cube2.position.y * rect.height;
              
              // Animated connection line
              const opacity = 0.3 + 0.2 * Math.sin(time * 0.01 + i + j);
              ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
              
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x2, y2);
              ctx.stroke();
            }
          }
        }
      }

      // Draw data flow particles
      if (isActive) {
        ctx.fillStyle = 'rgba(59, 130, 246, 0.6)';
        
        for (let i = 0; i < 5; i++) {
          const x = (0.1 + 0.8 * Math.sin(time * 0.005 + i)) * rect.width;
          const y = (0.1 + 0.8 * Math.cos(time * 0.003 + i)) * rect.height;
          const size = 2 + Math.sin(time * 0.01 + i);
          
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      time++;
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [cubeStates, isActive]);

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
        <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
        <span>Cube Network</span>
      </h3>

      {/* Canvas for connections and animations */}
      <div className="relative mb-6">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-80 pointer-events-none"
          style={{ width: '100%', height: '320px' }}
        />
        
        {/* Cube indicators */}
        <div className="relative w-full h-80">
          {Object.entries(cubeConfig).map(([cubeId, config]) => {
            const state = cubeStates[cubeId] || { active: false, load: 0, accuracy: 0 };
            
            return (
              <div
                key={cubeId}
                className="absolute transform -translate-x-1/2 -translate-y-1/2"
                style={{
                  left: `${config.position.x * 100}%`,
                  top: `${config.position.y * 100}%`
                }}
              >
                {/* Static Cube visualization */}
                <div>
                  <div
                    className={`w-16 h-16 rounded-xl border-2 flex items-center justify-center relative transition-all duration-500 ${
                      state.active 
                        ? 'border-white shadow-lg' 
                        : 'border-gray-500 border-opacity-50'
                    }`}
                    style={{
                      backgroundColor: state.active ? config.color : 'rgba(255, 255, 255, 0.1)',
                      boxShadow: state.active ? `0 0 30px ${config.color}60` : 'none'
                    }}
                  >
                    {/* Subtle activity indicator - only glow effect */}
                    {state.active && (
                      <motion.div
                        className="absolute inset-0 rounded-xl border-2 border-white border-opacity-30"
                        animate={{ opacity: [0.3, 0.7, 0.3] }}
                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                      />
                    )}
                    
                    {/* Static load indicator */}
                    <div className="text-white text-sm font-bold">
                      {state.active ? Math.round(state.load) : '0'}
                    </div>
                  </div>
                </div>

                {/* Static Cube label - NOT animated */}
                <div className="absolute left-1/2 transform -translate-x-1/2 text-center" style={{ top: '72px' }}>
                  <div className="text-white text-sm font-medium whitespace-nowrap mb-1">
                    {config.name}
                  </div>
                  <div className="text-gray-400 text-xs whitespace-nowrap mb-1">
                    {config.description}
                  </div>
                  {state.active && (
                    <div className="text-green-400 text-xs font-medium">
                      {(state.accuracy * 100).toFixed(0)}% acc
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Cube status list */}
      <div className="space-y-2">
        {Object.entries(cubeConfig).map(([cubeId, config]) => {
          const state = cubeStates[cubeId] || { active: false, load: 0, accuracy: 0 };
          
          return (
            <div
              key={cubeId}
              className={`flex items-center justify-between p-2 rounded-lg transition-all duration-200 ${
                state.active 
                  ? 'bg-white bg-opacity-10 border border-white border-opacity-20' 
                  : 'bg-gray-800 bg-opacity-30'
              }`}
            >
              <div className="flex items-center space-x-3">
                <div
                  className="w-4 h-4 rounded-md"
                  style={{ backgroundColor: state.active ? config.color : '#6b7280' }}
                />
                <span className={`text-sm font-medium ${state.active ? 'text-white' : 'text-gray-400'}`}>
                  {config.name}
                </span>
              </div>
              
              <div className="flex items-center space-x-2 text-xs">
                {state.active ? (
                  <>
                    <span className="text-gray-300">
                      Load: {Math.round(state.load)}%
                    </span>
                    <span className="text-green-400">
                      Acc: {(state.accuracy * 100).toFixed(0)}%
                    </span>
                  </>
                ) : (
                  <span className="text-gray-500">Inactive</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Network status */}
      <div className="mt-4 pt-4 border-t border-white border-opacity-20">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-300">Network Status:</span>
          <span className={`font-medium ${isActive ? 'text-green-400' : 'text-gray-500'}`}>
            {isActive ? 'Active Processing' : 'Standby'}
          </span>
        </div>
        
        {isActive && (
          <div className="flex items-center justify-between text-sm mt-1">
            <span className="text-gray-300">Active Cubes:</span>
            <span className="text-blue-400">
              {Object.values(cubeStates).filter(state => state.active).length} / 5
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default CubeVisualization;