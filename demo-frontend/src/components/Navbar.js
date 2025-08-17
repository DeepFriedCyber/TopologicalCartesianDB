import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Bars3Icon, 
  XMarkIcon,
  CubeIcon,
  ChartBarIcon,
  InformationCircleIcon,
  PlayIcon
} from '@heroicons/react/24/outline';
import { useDemo } from '../contexts/DemoContext';
import { useSocket } from '../contexts/SocketContext';

function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const { state } = useDemo();
  const { connected } = useSocket();

  const navigation = [
    { name: 'Home', href: '/', icon: CubeIcon },
    { name: 'Live Demo', href: '/demo', icon: PlayIcon },
    { name: 'Benchmarks', href: '/benchmark', icon: ChartBarIcon },
    { name: 'About', href: '/about', icon: InformationCircleIcon },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg border-b border-white border-opacity-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center"
              >
                <CubeIcon className="w-5 h-5 text-white" />
              </motion.div>
              <span className="text-xl font-bold text-white">
                Multi-Cube Demo
              </span>
            </Link>
          </div>

          {/* Desktop navigation */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 flex items-center space-x-1 ${
                      isActive(item.href)
                        ? 'bg-white bg-opacity-20 text-white'
                        : 'text-gray-300 hover:bg-white hover:bg-opacity-10 hover:text-white'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Status indicators */}
          <div className="hidden md:flex items-center space-x-4">
            {/* Connection status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-300">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {/* Session status */}
            {state.isSessionActive && (
              <div className="flex items-center space-x-2 bg-white bg-opacity-10 px-3 py-1 rounded-full">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm text-white">
                  Session Active
                </span>
              </div>
            )}

            {/* Context size indicator */}
            {state.sessionStats.context_size > 0 && (
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 px-3 py-1 rounded-full">
                <span className="text-sm text-white font-medium">
                  ~{(state.sessionStats.context_size / 1000).toFixed(0)}k tokens
                </span>
              </div>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="bg-white bg-opacity-10 inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-opacity-20 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
            >
              {isOpen ? (
                <XMarkIcon className="block h-6 w-6" />
              ) : (
                <Bars3Icon className="block h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <motion.div
        initial={false}
        animate={isOpen ? { height: 'auto', opacity: 1 } : { height: 0, opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="md:hidden overflow-hidden bg-white bg-opacity-10 backdrop-filter backdrop-blur-lg"
      >
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.href}
                onClick={() => setIsOpen(false)}
                className={`block px-3 py-2 rounded-md text-base font-medium transition-all duration-200 flex items-center space-x-2 ${
                  isActive(item.href)
                    ? 'bg-white bg-opacity-20 text-white'
                    : 'text-gray-300 hover:bg-white hover:bg-opacity-10 hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.name}</span>
              </Link>
            );
          })}
          
          {/* Mobile status indicators */}
          <div className="px-3 py-2 border-t border-white border-opacity-20 mt-4">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
                <span className="text-gray-300">
                  {connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              {state.isSessionActive && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  <span className="text-white">Session Active</span>
                </div>
              )}
            </div>
            
            {state.sessionStats.context_size > 0 && (
              <div className="mt-2">
                <span className="text-sm text-gray-300">
                  Context: ~{(state.sessionStats.context_size / 1000).toFixed(0)}k tokens
                </span>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </nav>
  );
}

export default Navbar;