import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Dashboard from './pages/Dashboard'
import './output.css';
import Onboarding from './pages/Onboarding';
import AppLayout from './layout/AppLayout';
import Personas from './pages/Personas'; 
import Zmot from './pages/Zmot'; 
import './output.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="/" element={<App />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/personas" element={<Personas />} />
          <Route path="/zmot" element={<Zmot />} />
          {/* Add other routes here */}
          

        </Routes>
      </AppLayout>
    </BrowserRouter>
  </React.StrictMode>
)
