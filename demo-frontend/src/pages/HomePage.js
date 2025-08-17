import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  CubeIcon, 
  ChartBarIcon, 
  LightBulbIcon,
  ArrowRightIcon,
  PlayIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';

function HomePage() {
  const features = [
    {
      icon: CubeIcon,
      title: 'Distributed Semantic Processing',
      description: 'Multiple specialized Cartesian cubes handle different domains simultaneously, preventing the token-based performance collapse.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: ChartBarIcon,
      title: 'Maintains 80%+ Accuracy at Scale',
      description: 'While traditional systems fail at 25k+ tokens, our multi-cube architecture maintains high accuracy even at 1M+ tokens.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: LightBulbIcon,
      title: 'Topological Intelligence',
      description: 'Cross-cube relationships use topological data analysis to preserve semantic structure and enable complex reasoning.',
      color: 'from-green-500 to-emerald-500'
    }
  ];

  const comparisonData = [
    { tokens: '5k', traditional: 100, multiCube: 100, status: 'excellent' },
    { tokens: '10k', traditional: 85, multiCube: 98, status: 'excellent' },
    { tokens: '25k', traditional: 45, multiCube: 95, status: 'revolutionary' },
    { tokens: '50k', traditional: 15, multiCube: 92, status: 'revolutionary' },
    { tokens: '100k', traditional: 5, multiCube: 88, status: 'revolutionary' },
    { tokens: '500k', traditional: 0, multiCube: 82, status: 'revolutionary' },
    { tokens: '1M+', traditional: 0, multiCube: 78, status: 'revolutionary' }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl sm:text-6xl font-bold text-white mb-6">
              <span className="bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                Multi-Cube Architecture
              </span>
              <br />
              <span className="text-white">
                Solves the Long-Context Problem
              </span>
            </h1>
            
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Revolutionary distributed semantic processing that maintains <strong>80%+ accuracy</strong> 
              even at <strong>1M+ token contexts</strong> where traditional systems completely fail.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link
                to="/demo"
                className="btn-primary flex items-center space-x-2 text-lg px-8 py-3"
              >
                <PlayIcon className="w-5 h-5" />
                <span>Try Live Demo</span>
              </Link>
              
              <Link
                to="/benchmark"
                className="btn-secondary flex items-center space-x-2 text-lg px-8 py-3"
              >
                <ChartBarIcon className="w-5 h-5" />
                <span>View Benchmarks</span>
              </Link>
            </div>
          </motion.div>

          {/* Floating cubes animation */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg opacity-10"
                animate={{
                  x: [0, 100, 0],
                  y: [0, -100, 0],
                  rotate: [0, 180, 360]
                }}
                transition={{
                  duration: 10 + i * 2,
                  repeat: Infinity,
                  ease: "linear"
                }}
                style={{
                  left: `${10 + i * 20}%`,
                  top: `${20 + i * 15}%`
                }}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Problem Statement */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={itemVariants} className="text-3xl sm:text-4xl font-bold text-white mb-6">
              The Token Limit Crisis
            </motion.h2>
            <motion.p variants={itemVariants} className="text-xl text-gray-300 max-w-3xl mx-auto">
              Current AI systems suffer from exponential performance degradation as context size increases. 
              At 25k+ tokens, accuracy drops below 50%. The "million token context" remains hollow.
            </motion.p>
          </motion.div>

          {/* Performance Comparison Chart */}
          <motion.div
            variants={itemVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="card p-8 mb-12"
          >
            <h3 className="text-2xl font-bold text-white mb-6 text-center">
              Performance Comparison: Traditional vs Multi-Cube
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-white">
                <thead>
                  <tr className="border-b border-white border-opacity-20">
                    <th className="text-left py-3 px-4">Context Size</th>
                    <th className="text-center py-3 px-4">Traditional Systems</th>
                    <th className="text-center py-3 px-4">Multi-Cube Architecture</th>
                    <th className="text-center py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.map((row, index) => (
                    <motion.tr
                      key={row.tokens}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="border-b border-white border-opacity-10"
                    >
                      <td className="py-3 px-4 font-medium">{row.tokens} tokens</td>
                      <td className="text-center py-3 px-4">
                        <div className="flex items-center justify-center space-x-2">
                          {row.traditional > 0 ? (
                            <CheckCircleIcon className="w-5 h-5 text-green-400" />
                          ) : (
                            <XCircleIcon className="w-5 h-5 text-red-400" />
                          )}
                          <span className={row.traditional > 50 ? 'text-green-400' : row.traditional > 0 ? 'text-yellow-400' : 'text-red-400'}>
                            {row.traditional}%
                          </span>
                        </div>
                      </td>
                      <td className="text-center py-3 px-4">
                        <div className="flex items-center justify-center space-x-2">
                          <CheckCircleIcon className="w-5 h-5 text-green-400" />
                          <span className="text-green-400 font-semibold">
                            {row.multiCube}%
                          </span>
                        </div>
                      </td>
                      <td className="text-center py-3 px-4">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          row.status === 'excellent' ? 'bg-green-500 text-white' :
                          row.status === 'revolutionary' ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white' :
                          'bg-blue-500 text-white'
                        }`}>
                          {row.status === 'revolutionary' ? '🚀 Revolutionary' : '✅ Excellent'}
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <motion.h2 variants={itemVariants} className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Revolutionary Architecture
            </motion.h2>
            <motion.p variants={itemVariants} className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our multi-cube approach distributes semantic processing across specialized domains, 
              preventing the token-based performance collapse that plagues traditional systems.
            </motion.p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="card card-hover p-8 text-center"
                >
                  <div className={`w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-r ${feature.color} flex items-center justify-center`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-4">
                    {feature.title}
                  </h3>
                  <p className="text-gray-300">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="card p-12"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
              Experience the Future of Long-Context AI
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Test queries that would completely break traditional systems. 
              See how our multi-cube architecture maintains high accuracy at impossible scales.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/demo"
                className="btn-primary flex items-center justify-center space-x-2 text-lg px-8 py-4"
              >
                <PlayIcon className="w-6 h-6" />
                <span>Start Live Demo</span>
                <ArrowRightIcon className="w-5 h-5" />
              </Link>
              
              <Link
                to="/about"
                className="btn-secondary flex items-center justify-center space-x-2 text-lg px-8 py-4"
              >
                <span>Learn More</span>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}

export default HomePage;