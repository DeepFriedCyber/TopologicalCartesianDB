import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { DemoProvider } from './contexts/DemoContext';
import { SocketProvider } from './contexts/SocketContext';

// Components
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import DemoPage from './pages/DemoPage';
import BenchmarkPage from './pages/BenchmarkPage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <div className="App min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      <Router>
        <SocketProvider>
          <DemoProvider>
            <Navbar />
            <main className="pt-16">
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/demo" element={<DemoPage />} />
                <Route path="/benchmark" element={<BenchmarkPage />} />
                <Route path="/about" element={<AboutPage />} />
              </Routes>
            </main>
            <Toaster 
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  color: 'white',
                },
              }}
            />
          </DemoProvider>
        </SocketProvider>
      </Router>
    </div>
  );
}

export default App;